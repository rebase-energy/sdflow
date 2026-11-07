"""One engine, three modes: a live session replays a backtest exactly, survives being
persisted and restored, and simulation runs the same loop over sampled data."""

import numpy as np
import pandas as pd
import pytest

import sdflow as sd
from sdflow.models import Agent
from sdflow.run.causality import truncate_dataset
from sdflow.run.live import FixedClock, LiveSession
from sdflow.run.simulate import simulate

from test_agent import LastValue, ThresholdStorage, YesterdayPrice
from test_decision_env import make_problem


def replay(problem, model, split="validation", limit=None):
    """Drive a LiveSession through the split's origins on truncated data, then
    flush with the full dataset; return (actions by asof, settled records, session)."""
    full = problem.load_dataset()
    origins = problem.origins(split)[:limit]
    session = LiveSession(problem, model, clock=FixedClock(origins[0].asof))
    actions, settled = {}, []
    for origin in origins:
        step = session.step(truncate_dataset(full, origin.asof), now=origin.asof)
        assert step.origin is not None and step.origin.asof == origin.asof
        actions[origin.asof] = step.action
        settled.extend(step.settled)
    # the world moves on: the remaining pending decisions become knowable
    flush = max(o.target_end for o in origins) + pd.Timedelta(days=2)
    step = session.step(full, now=flush)
    assert step.origin is None or step.origin.asof > origins[-1].asof
    settled.extend(step.settled)
    return actions, settled, session


def backtest_actions(result):
    return {asof: block.droplevel("asof") for asof, block in result.predictions.groupby(level="asof")}


class TestLiveReplaysBacktest:
    def test_stateless(self, toy_problem):
        result = sd.Experiment(toy_problem, LastValue(), analyzers=None).run()
        actions, settled, _ = replay(toy_problem, LastValue(), limit=40)
        expected = backtest_actions(result)
        for asof, action in actions.items():
            pd.testing.assert_frame_equal(action, expected[asof], check_exact=True, check_names=False)
        scores = pd.Series({r.origin.asof: r.score for r in settled}).sort_index()
        pd.testing.assert_series_equal(
            scores, result.settlements["score"].loc[scores.index], check_names=False)

    def test_stateful(self):
        problem = make_problem()
        make = lambda: Agent(YesterdayPrice(), ThresholdStorage(k=2))  # noqa: E731
        result = sd.Experiment(problem, make(), analyzers=None).run()
        actions, settled, session = replay(problem, make())
        expected = backtest_actions(result)
        assert actions.keys() == expected.keys()
        for asof, action in actions.items():
            pd.testing.assert_frame_equal(action, expected[asof], check_exact=True, check_names=False)
        live_scores = pd.Series({r.origin.asof: r.score for r in settled}).sort_index()
        pd.testing.assert_series_equal(live_scores, result.settlements["score"], check_names=False)
        # the flush wake-up opened a later origin, so only that decision may still be pending
        assert all(o.asof > problem.origins("validation")[-1].asof for o, _ in session.pending)
        assert np.isclose(session.state.level, settled[-1].state.level)


class TestSessionMemory:
    def test_restore_continues_identically(self):
        problem = make_problem()
        full = problem.load_dataset()
        origins = problem.origins("validation")
        make = lambda: Agent(YesterdayPrice(), ThresholdStorage(k=2))  # noqa: E731

        whole = LiveSession(problem, make(), clock=FixedClock(origins[0].asof))
        uninterrupted = [whole.step(truncate_dataset(full, o.asof), now=o.asof).action for o in origins]

        model = make()
        first = LiveSession(problem, model, clock=FixedClock(origins[0].asof))
        for o in origins[:5]:
            first.step(truncate_dataset(full, o.asof), now=o.asof)
        memory = dict(state=first.state, pending=first.pending, last_origin=first.last_origin)

        second = LiveSession(problem, model, clock=FixedClock(origins[5].asof)).restore(**memory)
        resumed = [second.step(truncate_dataset(full, o.asof), now=o.asof).action for o in origins[5:]]
        for a, b in zip(uninterrupted[5:], resumed):
            pd.testing.assert_frame_equal(a, b, check_exact=True)

    def test_nothing_due_between_origins(self):
        problem = make_problem()
        full = problem.load_dataset()
        origins = problem.origins("validation")
        session = LiveSession(problem, Agent(YesterdayPrice(), ThresholdStorage()))
        session.step(truncate_dataset(full, origins[0].asof), now=origins[0].asof)
        between = origins[0].asof + pd.Timedelta("3h")
        step = session.step(truncate_dataset(full, between), now=between)
        assert step.origin is None and step.action is None and step.settled == []
        assert len(session.pending) == 1

    def test_training_happens_once_from_the_training_view(self, toy_problem):
        pred = LastValue()
        session = LiveSession(toy_problem, pred)
        full = toy_problem.load_dataset()
        o = toy_problem.origins("validation")[0]
        session.step(truncate_dataset(full, o.asof), now=o.asof)
        assert pred.fitted and session.started
        pred.fitted = False
        o2 = toy_problem.origins("validation")[1]
        session.step(truncate_dataset(full, o2.asof), now=o2.asof)
        assert not pred.fitted, "a live session fits once, then only learns"


class TestSimulate:
    def test_signal_game_over_scenarios(self):
        from sdflow.benchmarks.signal_game.baseline import BayesDig
        from sdflow.benchmarks.signal_game.problem import task_for

        problem = sd.load_problem("signal-game:w8")
        task = task_for("w8")
        sim = simulate(problem, lambda: BayesDig(task), n_scenarios=3, seed=11)
        assert len(sim.results) == 3 and list(sim.seeds) == [11, 12, 13]
        assert sim.score == pytest.approx(0.9 ** 3, abs=0.05)
        assert sim.spread < 0.1 and sim.model == "bayes"
        assert len(set(round(s, 6) for s in sim.scores)) > 1, "scenarios differ"

    def test_history_only_problem_cannot_be_simulated(self, toy_problem):
        with pytest.raises(ValueError, match="no sampler"):
            toy_problem.scenario(0)
