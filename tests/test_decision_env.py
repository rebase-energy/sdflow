"""DecisionEnv: state carried by a Simulator, settlement pluggable, ForecastEnv the
stateless case. The toy is storage arbitrage: an hourly price, a store with a
capacity and a rate limit, a day-ahead schedule of flows settled at the realised
price once it is knowable."""

import numpy as np
import pandas as pd
import pytest

import sdflow as sd
from sdflow.data import DataFeed, Dataset, Field
from sdflow.envs import (
    ColumnsAction, DecisionEnv, ForecastEnv, ScoreSettlement, Settlement, SettlementContext,
)
from sdflow.models import Predictor, StorageSimulator, StorageState
from sdflow.problems import IssueSchedule, Metric, Objective, Problem, Splits

TRAIN_END = pd.Timestamp("2025-01-20", tz="UTC")
VAL = (pd.Timestamp("2025-01-20", tz="UTC"), pd.Timestamp("2025-02-01", tz="UTC"))
HOLDOUT = (pd.Timestamp("2025-02-01", tz="UTC"), pd.Timestamp("2025-02-10", tz="UTC"))
CAPACITY, RATE = 10.0, 5.0


def price_dataset(seed=0) -> Dataset:
    idx = pd.date_range("2025-01-01", "2025-02-15", freq="1h", tz="UTC")
    rng = np.random.default_rng(seed)
    hour = idx.hour.to_numpy()
    price = 50.0 + 20.0 * np.sin(2 * np.pi * (hour - 6) / 24) + rng.normal(0, 2.0, len(idx))
    frame = pd.DataFrame({"price": price}, index=idx)
    # the realised price is published one hour after delivery
    return Dataset(name="prices", fields={"price": Field("price", frame, availability_lag="1h")})


class ArbitrageRevenue(Metric):
    aggregate = "sum"
    settled_by_env = True

    def elementwise(self, y_true, y_pred):
        raise NotImplementedError("settled by the environment")


class StorageRevenue(Settlement):
    """Money made: sell (negative flow) earns, buy (positive flow) costs, priced
    on the flows the store actually executed."""

    fields = ("price",)

    def settle(self, ctx: SettlementContext) -> float:
        return float(-(ctx.series("price") * ctx.state_after.realized).sum())


def make_problem(schedule=None) -> Problem:
    dataset = price_dataset()
    schedule = schedule or IssueSchedule.daily(at="12:00", covers=("12h", "35h"), target_freq="1h")

    def make_env(problem, split):
        return DecisionEnv(
            feed=DataFeed(problem.load_dataset()),
            origins=problem.origins(split),
            settlement=StorageRevenue(),
            objective=problem.objective,
            action_spec=ColumnsAction("flow"),
            simulator=StorageSimulator(CAPACITY, max_inflow=RATE, max_outflow=RATE, column="flow"),
            train_end=problem.splits.train_end,
        )

    return Problem(
        name="toy:storage",
        dataset=dataset,
        make_env=make_env,
        objective=Objective(ArbitrageRevenue(), lower_is_better=False),
        schedule=schedule,
        splits=Splits(train_end=TRAIN_END, validation=VAL, holdout=HOLDOUT),
    )


class YesterdayArbitrage(Predictor):
    """Buy at full rate in yesterday's two cheapest hours, sell in its two dearest."""

    def predict(self, obs) -> pd.DataFrame:
        last_day = obs.history("price")["price"].dropna().iloc[-24:]
        by_hour = last_day.groupby(last_day.index.hour).mean().sort_values()
        cheap, dear = set(by_hour.index[:2]), set(by_hour.index[-2:])
        hours = obs.target_index.hour
        flow = np.where(np.isin(hours, list(cheap)), RATE,
                        np.where(np.isin(hours, list(dear)), -RATE, 0.0))
        return pd.DataFrame({"flow": flow}, index=obs.target_index)


def run_env(problem, agent, split="validation"):
    env = problem.env(split)
    obs, info = env.reset()
    observations, records = [obs], []
    while obs is not None:
        obs, reward, terminated, _, info = env.step(agent.predict(obs))
        records.extend(info["settled"])
        if obs is not None:
            observations.append(obs)
    return env, observations, records


@pytest.fixture
def problem():
    return make_problem()


class TestState:
    def test_initial_state_comes_from_the_simulator(self, problem):
        obs, _ = problem.env().reset()
        assert obs.state == StorageState(0.0)
        assert obs.state.realized is None and obs.pending == ()

    def test_state_is_carried_between_origins_and_bounded(self, problem):
        _, observations, records = run_env(problem, YesterdayArbitrage())
        levels = [o.state.level for o in observations]
        assert len(set(levels)) > 1, "the level never moved"
        assert all(0.0 <= lvl <= CAPACITY for lvl in levels)
        # the state an observation carries is the state after the last settlement
        assert observations[-1].state == records[-2].state or observations[-1].state == records[-1].state

    def test_pending_commitments_are_visible_until_settled(self, problem):
        """A 12:00 origin covering the next day settles only once the price for
        23:00 the next day is knowable (00:00 + 1h), i.e. two origins later."""
        _, observations, _ = run_env(problem, YesterdayArbitrage())
        assert len(observations[1].pending) == 1
        assert len(observations[2].pending) == 1   # yesterday's settled, today's pending
        assert observations[1].pending[0][0].asof == observations[0].asof

    def test_simulator_clips_flows_to_rate_and_capacity(self, problem):
        _, _, records = run_env(problem, YesterdayArbitrage())
        level = 0.0
        for r in records:
            realized = r.state.realized
            assert (realized.abs() <= RATE + 1e-9).all()
            level = level + realized.sum()
            assert -1e-9 <= level <= CAPACITY + 1e-9
            assert np.isclose(level, r.state.level)

    def test_settlement_prices_the_realized_flows(self, problem):
        env, _, records = run_env(problem, YesterdayArbitrage())
        r = records[0]
        price = env.feed.dataset.field("price").frame["price"].reindex(r.origin.target_index)
        assert np.isclose(r.score, -(price * r.state.realized).sum())
        assert r.actuals is None and r.n_scored == 24


class TestSettlementTiming:
    def test_waits_for_the_slowest_field(self):
        """Hourly one-step decisions: the price for hour t is knowable at t+1h,
        so an origin cannot settle at the very next origin."""
        problem = make_problem(IssueSchedule.hourly(horizon="1h"))
        env = problem.env()
        obs, _ = env.reset()
        obs, _, _, _, info = env.step(pd.DataFrame({"flow": 1.0}, index=obs.target_index))
        assert info["settled"] == []
        obs, _, _, _, info = env.step(pd.DataFrame({"flow": 1.0}, index=obs.target_index))
        assert len(info["settled"]) == 1

    def test_stateful_env_settles_in_origin_order(self, problem):
        _, _, records = run_env(problem, YesterdayArbitrage())
        asofs = [r.origin.asof for r in records]
        assert asofs == sorted(asofs)
        assert len(records) == len(problem.origins("validation"))


class TestExperiment:
    def test_event_mode_sums_revenue(self, problem):
        result = sd.Experiment(problem, YesterdayArbitrage(), analyzers=None).run()
        assert result.mode == "event" and result.aggregate == "sum"
        assert np.isclose(result.score, result.settlements["score"].sum())
        assert (result.settlements["n_scored"] == 24).all()
        assert result.score > 0, "buying cheap and selling dear should pay"

    def test_vectorized_mode_is_refused_for_stateful_envs(self, problem):
        class Batchy(YesterdayArbitrage):
            supports_batch = True

        with pytest.raises(ValueError, match="cannot see the state"):
            sd.Experiment(problem, Batchy(), analyzers=None).run(mode="vectorized")
        assert sd.Experiment(problem, Batchy(), analyzers=None).run().mode == "event"

    def test_default_analyzers_cope_without_a_target_series(self, problem):
        result = sd.Experiment(problem, YesterdayArbitrage()).run()
        assert result.analysis == {}


class TestForecastEnvIsTheStatelessCase:
    def test_forecast_env_composition(self, toy_problem):
        env = toy_problem.env()
        assert isinstance(env, DecisionEnv) and not env.stateful
        assert isinstance(env.settlement, ScoreSettlement)
        obs, _ = env.reset()
        assert obs.state is None

    def test_settlement_is_pluggable_on_forecast_env(self, toy_problem):
        class Constant(Settlement):
            fields = ("power",)

            def settle(self, ctx):
                return 1.0

        base = toy_problem.env()
        env = ForecastEnv(
            feed=base.feed, origins=base.origins, target_field="power",
            objective=toy_problem.objective, settlement=Constant(),
        )
        obs, _ = env.reset()
        _, reward, _, _, info = env.step(pd.DataFrame({"point": 0.0}, index=obs.target_index))
        assert [r.score for r in info["settled"]] == [1.0]
        assert reward == -1.0   # MAE objective is lower-is-better
        assert info["settled"][0].actuals is None and info["settled"][0].n_scored == 1
