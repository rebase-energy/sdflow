"""The signal game is a measurement against known truth: Bayes reaches its exact
expected payoff, the learner gets there from data, chance gets 1/W, and every
agent is truncation invariant."""

import numpy as np
import pytest

import sdflow as sd
from sdflow.benchmarks.signal_game.baseline import BayesDig, ChanceDig, LearnedDig
from sdflow.benchmarks.signal_game.problem import (
    build_dataset, build_task, get_problem_w8, task_for,
)
from sdflow.run.causality import assert_truncation_invariant


@pytest.fixture(scope="module")
def problem():
    return get_problem_w8()


@pytest.fixture(scope="module")
def task():
    return task_for("w8")


def ref(problem, team_prefix):
    return next(r.score for r in problem.reference_scores if r.team.startswith(team_prefix))


class TestTask:
    def test_tables(self, task):
        assert task.likelihood.shape == (6, 8, 4) and task.payoff.shape == (8, 8)
        np.testing.assert_allclose(task.likelihood.sum(axis=2), 1.0)
        assert sorted(task.labels) == ["bit0", "bit1", "bit2", "noise", "noise", "noise"]
        assert task.chance_payoff() == pytest.approx(1 / 8)
        assert task.max_payoff() == pytest.approx(1.0)

    def test_bayes_value_is_exact_and_between_chance_and_oracle(self, task):
        v = task.bayes_payoff()
        assert 1 / 8 < v < 1.0
        # three independent bits each right with prob 0.9 (noise sensors add nothing):
        assert v == pytest.approx(0.9 ** 3, abs=1e-9)

    def test_exact_value_matches_chunked_enumeration_for_w16(self):
        task = build_task(n_places=16, n_signal=6, n_noise=4, flip=0.2)
        # bits 0,1 are read twice (sensors 0..5 cover bits 0,1,2,3,0,1); exact by hand:
        # a bit read twice with flip ε is right if both agree on truth (prob (1-ε)²),
        # tied otherwise; with ties broken at argmax the value sits between the bounds.
        v = task.bayes_payoff()
        assert 1 / 16 < v < 1.0
        single, double = 0.8, 0.8 ** 2 + 0.5 * 2 * 0.8 * 0.2
        assert v == pytest.approx(single ** 2 * double ** 2, abs=1e-9)

    def test_dataset_is_deterministic_and_well_stamped(self, task):
        a, b = build_dataset(task, 50, seed=0), build_dataset(task, 50, seed=0)
        assert a.fields["sensors"].frame.equals(b.fields["sensors"].frame)
        sensors, hidden = a.fields["sensors"].frame, a.fields["hidden"].frame
        assert len(sensors) == len(hidden) == 50
        assert ((hidden.index - sensors.index) == np.timedelta64(1, "h")).all()
        assert sensors.to_numpy().min() >= 0 and sensors.to_numpy().max() <= 3


class TestAgents:
    def test_registered(self):
        assert "signal-game:w8" in sd.list_problems() and "signal-game:w16" in sd.list_problems()

    def test_bayes_reaches_its_exact_value(self, problem, task):
        result = sd.Experiment(problem, BayesDig(task), analyzers=None).run()
        assert result.n_origins == 500
        # 500 Bernoulli trials around 0.729: three standard errors is 0.06
        assert result.score == pytest.approx(ref(problem, "bayes"), abs=0.06)
        assert problem.rank_of(result.score) in (2, 3)

    def test_chance_scores_the_prior(self, problem):
        result = sd.Experiment(problem, ChanceDig(task_for("w8").n_places), analyzers=None).run()
        assert result.score == pytest.approx(1 / 8, abs=0.05)

    def test_learner_gets_to_bayes_from_data(self, problem, task):
        learner = LearnedDig(task.n_places, task.n_sensors, task.n_symbols)
        result = sd.Experiment(problem, learner, analyzers=None).run()
        bayes = sd.Experiment(problem, BayesDig(task), analyzers=None).run()
        assert result.score == pytest.approx(bayes.score, abs=0.05)
        # it found the channel structure: noise sensors look flat, signal sensors peaked
        peak = learner.likelihood.max(axis=2).mean(axis=1)
        signal = [k for k, lbl in enumerate(task.labels) if lbl != "noise"]
        noise = [k for k, lbl in enumerate(task.labels) if lbl == "noise"]
        assert peak[signal].min() > 0.8 and peak[noise].max() < 0.45

    def test_online_learning_alone_improves(self, problem, task):
        class Cold(LearnedDig):
            def fit(self, train):
                return self      # no training view: learn only from settled episodes

        learner = Cold(task.n_places, task.n_sensors, task.n_symbols)
        result = sd.Experiment(problem, learner, analyzers=None).run()
        scores = result.settlements["score"]
        assert scores.iloc[:50].mean() < scores.iloc[-100:].mean()
        assert scores.iloc[-100:].mean() > 0.55


class TestFrameworkSelfCheck:
    def test_every_baseline_is_truncation_invariant(self, problem, task):
        for make in (lambda: BayesDig(task),
                     lambda: LearnedDig(task.n_places, task.n_sensors, task.n_symbols),
                     lambda: ChanceDig(task.n_places)):
            assert assert_truncation_invariant(problem, make, k=4).ok

    def test_an_episode_settles_when_the_next_origin_opens(self, problem, task):
        """The place is stamped at the target time with no lag, and the next origin
        opens exactly then, so the dig submitted in a step is settled in that step."""
        env = problem.env()
        obs, _ = env.reset()
        agent = BayesDig(task)
        first = obs.asof
        obs, reward, _, _, info = env.step(agent.act(obs))
        assert len(info["settled"]) == 1 and info["settled"][0].origin.asof == first
        assert reward in (0.0, 1.0)
        assert obs.history("hidden").index.max() == info["settled"][0].origin.target_end
