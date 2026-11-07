"""A problem declares its own analyzers; RegimeBreakdown splits score and bias by a post-hoc regime."""

import numpy as np
import pandas as pd

import sdflow as sd
from sdflow.data import Field

from conftest import make_toy_problem


class Constant(sd.Predictor):
    """Predicts a constant, so the bias per regime is known exactly."""

    def __init__(self, value: float) -> None:
        super().__init__("constant")
        self.value = value

    def fit(self, train):
        return self

    def predict(self, obs):
        return pd.DataFrame({"point": self.value}, index=obs.target_index)


def _with_regime(problem: sd.Problem) -> sd.Problem:
    dataset = problem.load_dataset()
    power = dataset.fields["power"].frame
    # Regime: the top third of the day is "high" (hours 6..13 of the sine).
    hot = pd.DataFrame({"temperature": np.where(power.index.hour.isin(range(6, 14)), 30.0, 5.0)}, index=power.index)
    dataset.fields["weather"] = Field("weather", hot, availability_lag="0h")
    problem.analyzers = [lambda: sd.RegimeBreakdown("weather", "temperature", threshold=20.0, labels=("hot", "cold"))]
    return problem


def test_declared_analyzer_runs_with_the_defaults_and_splits_the_score() -> None:
    problem = _with_regime(make_toy_problem())
    result = sd.evaluate(problem, Constant(10.0), split="validation")
    stats = result.stats()
    assert "PersistenceSkill.model_score" in stats, "defaults still run"
    for key in ("hot_score", "cold_score", "hot_hours", "cold_hours", "hot_bias", "cold_bias", "hot_share"):
        assert f"RegimeBreakdown.{key}" in stats, key
    hot, cold = stats["RegimeBreakdown.hot_hours"], stats["RegimeBreakdown.cold_hours"]
    assert hot + cold == result.n_scored
    assert abs(stats["RegimeBreakdown.hot_share"] - 8 / 24) < 0.05
    # The sine is above 10 in the hot hours (6..13) and the constant predicts 10: positive bias there.
    assert stats["RegimeBreakdown.hot_bias"] > 0 > stats["RegimeBreakdown.cold_bias"]
    # The per-regime scores average back to the overall MAE, weighted by hours.
    overall = (stats["RegimeBreakdown.hot_score"] * hot + stats["RegimeBreakdown.cold_score"] * cold) / (hot + cold)
    assert abs(overall - result.score) < 1e-9


def test_explicit_analyzers_replace_the_declared_ones() -> None:
    problem = _with_regime(make_toy_problem())
    result = sd.Experiment(problem, Constant(10.0), analyzers=[]).run()
    assert result.analysis == {}


def test_bins_give_several_regimes() -> None:
    problem = make_toy_problem()
    dataset = problem.load_dataset()
    power = dataset.fields["power"].frame
    dataset.fields["load"] = Field("load", pd.DataFrame({"level": power.index.hour}, index=power.index), availability_lag="0h")
    problem.analyzers = [lambda: sd.RegimeBreakdown("load", "level", bins=[0, 8, 16, 24], labels=("night", "day", "evening"))]
    stats = sd.evaluate(problem, Constant(10.0), split="validation").stats()
    assert {f"RegimeBreakdown.{k}_hours" for k in ("night", "day", "evening")} <= set(stats)
