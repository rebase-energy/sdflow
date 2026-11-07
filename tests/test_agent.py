"""Agent = Predictor ∘ Optimizer: the same predictor scores on a forecast problem and,
behind an optimizer, drives a stateful decision problem."""

import numpy as np
import pandas as pd
import pytest

import sdflow as sd
from sdflow.models import (
    Agent, Decision, IdentityOptimizer, Optimizer, Predictor, QuantileDecision,
    decision, forecast, point_of, quantile_levels, quantile_of,
)
from sdflow.run import Verifier, load_submission

from test_decision_env import RATE, make_problem


class LastValue(Predictor):
    def predict(self, obs):
        last = obs.history(self.field)[self.field].dropna().iloc[-1]
        return forecast(obs.target_index, point=last)

    def __init__(self, field="power"):
        super().__init__()
        self.field = field
        self.fitted = False

    def fit(self, train):
        self.fitted = True
        return self


class YesterdayPrice(Predictor):
    """Point forecast of the price: the same hour yesterday."""

    def predict(self, obs):
        hist = obs.history("price")["price"].dropna().iloc[-24:]
        by_hour = hist.groupby(hist.index.hour).mean()
        return forecast(obs.target_index, point=by_hour.reindex(obs.target_index.hour).to_numpy())


class ThresholdStorage(Optimizer):
    """Buy at full rate in the k cheapest forecast hours, sell in the k dearest,
    never selling more than the store holds."""

    def __init__(self, k=2):
        super().__init__()
        self.k = k

    def decide(self, fc, obs) -> Decision:
        price = point_of(fc)
        order = np.argsort(price.to_numpy())
        flow = np.zeros(len(price))
        flow[order[: self.k]] = RATE
        flow[order[-self.k:]] = -min(RATE, max(obs.state.level, 0.0) + RATE * self.k / 2)
        return decision(obs.target_index, flow=flow)


class TestFrames:
    def test_forecast_builders_and_accessors(self):
        idx = pd.date_range("2025-01-01", periods=3, freq="1h", tz="UTC")
        fc = forecast(idx, quantiles={0.1: [1, 1, 1], 0.5: [2, 2, 2], 0.9: [4, 4, 4]})
        assert quantile_levels(fc) == [0.1, 0.5, 0.9]
        assert (point_of(fc) == 2).all()                     # the median
        assert (quantile_of(fc, 0.7) == 3).all()             # interpolated
        assert (quantile_of(fc, 0.99) == 4).all()            # clamped to the nearest level
        pt = forecast(idx, point=[5, 5, 5])
        assert quantile_levels(pt) == [] and (quantile_of(pt, 0.2) == 5).all()
        with pytest.raises(ValueError):
            forecast(idx)
        with pytest.raises(ValueError):
            decision(idx)

    def test_quantile_decision_is_the_newsvendor_fractile(self):
        idx = pd.date_range("2025-01-01", periods=2, freq="1h", tz="UTC")
        fc = forecast(idx, quantiles={0.25: [10, 10], 0.75: [30, 30]})
        opt = QuantileDecision(c_under=3, c_over=1)          # level 0.75
        d = opt.decide(fc, obs=None)
        assert opt.level == 0.75 and list(d.columns) == ["bid"] and (d["bid"] == 30).all()
        with pytest.raises(ValueError):
            QuantileDecision()


class TestAgentOnForecastProblem:
    def test_identity_agent_matches_the_bare_predictor(self, toy_problem):
        bare = sd.Experiment(toy_problem, LastValue(), analyzers=None).run()
        agent = Agent(LastValue(), IdentityOptimizer())
        composed = sd.Experiment(toy_problem, agent, analyzers=None).run()
        assert composed.score == pytest.approx(bare.score)
        assert composed.model == "LastValue → IdentityOptimizer"

    def test_fit_and_bind_reach_the_predictor(self, toy_problem):
        pred = LastValue()
        agent = Agent(pred)
        sd.Experiment(toy_problem, agent, analyzers=None).run()
        assert pred.fitted and pred.problem is toy_problem and agent.problem is toy_problem

    def test_learn_receives_every_settlement(self, toy_problem):
        seen = []

        class Watching(Agent):
            def learn(self, record):
                seen.append(record.origin.asof)

        result = sd.Experiment(toy_problem, Watching(LastValue()), analyzers=None).run()
        assert len(seen) == result.n_origins

    def test_agent_needs_a_part_or_its_own_act(self):
        with pytest.raises(ValueError, match="needs a predictor"):
            Agent()

        class Rule(Agent):
            def act(self, obs):
                return forecast(obs.target_index, point=0.0)

        assert Rule().name == "Rule"


class TestAgentOnStatefulProblem:
    def test_forecast_flows_into_the_optimizer_and_state_is_used(self):
        problem = make_problem()
        agent = Agent(YesterdayPrice(), ThresholdStorage(k=2))
        result = sd.Experiment(problem, agent, analyzers=None).run()
        assert result.mode == "event" and result.n_origins == len(problem.origins("validation"))
        assert result.score > 0
        assert (result.settlements["n_scored"] == 24).all()

    def test_same_predictor_two_problems(self, toy_problem):
        """One predictor class, a forecast leaderboard and a decision leaderboard."""
        problem = make_problem()
        fc_result = sd.Experiment(toy_problem, LastValue("power"), analyzers=None).run()
        dec_result = sd.Experiment(problem, Agent(YesterdayPrice(), ThresholdStorage()),
                                   analyzers=None).run()
        assert fc_result.objective != dec_result.objective
        assert np.isfinite(fc_result.score) and np.isfinite(dec_result.score)


class TestIntegrationSurfaces:
    def test_verifier_accepts_an_agent(self, toy_problem):
        v = Verifier(toy_problem, leaderboard_path=None)
        result = v.verify(lambda: Agent(LastValue(), IdentityOptimizer()), verbose=False)
        assert np.isfinite(result.score) and result.split == "holdout"

    def test_submission_module_may_expose_an_agent(self, tmp_path):
        mod = tmp_path / "sub.py"
        mod.write_text(
            "import pandas as pd\n"
            "from sdflow.models import Agent, IdentityOptimizer, Predictor, forecast\n"
            "class P(Predictor):\n"
            "    def predict(self, obs):\n"
            "        return forecast(obs.target_index, point=0.0)\n"
            "model = Agent(P(), IdentityOptimizer())\n"
        )
        loaded = load_submission(mod)
        assert isinstance(loaded, Agent)
