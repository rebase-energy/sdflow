"""The truncation-invariance check catches a model that reads the future and
passes a clean one, on stateless and stateful problems."""

import pytest

from sdflow.models import Agent, Predictor, forecast
from sdflow.run.causality import (
    assert_truncation_invariant, truncate_dataset, truncation_invariance,
)

from test_agent import ThresholdStorage, YesterdayPrice
from test_decision_env import make_problem


class Clean(Predictor):
    def predict(self, obs):
        last = obs.history("power")["power"].dropna().iloc[-1]
        return forecast(obs.target_index, point=last)


class Peeks(Predictor):
    """Reads the raw dataset behind the feed: the actual at the target time."""

    def predict(self, obs):
        raw = obs._feed.dataset.field("power").frame["power"]
        value = raw.reindex(obs.target_index).fillna(raw.iloc[-1])
        return forecast(obs.target_index, point=value.to_numpy())


class Remembers(Predictor):
    """Leaks through its own memory: carries a running mean across the whole run
    computed from data it was handed later than the origin (the training view
    is fine; the raw field is not)."""

    def fit(self, train):
        self.mean = float(train._feed.dataset.field("power").frame["power"].mean())
        return self

    def predict(self, obs):
        return forecast(obs.target_index, point=self.mean)


def test_truncate_dataset_removes_the_future(toy_problem):
    ds = toy_problem.load_dataset()
    asof = toy_problem.origins("validation")[10].asof
    cut = truncate_dataset(ds, asof)
    assert cut.fields["power"].frame.index.max() <= asof
    issues = cut.fields["nwp"].frame.index.get_level_values("issue_time")
    assert (issues + cut.fields["nwp"].availability_lag).max() <= asof
    assert len(cut.fields["power"].frame) < len(ds.fields["power"].frame)


def test_clean_predictor_passes(toy_problem):
    report = assert_truncation_invariant(toy_problem, Clean, k=4)
    assert report.ok and len(report.checked) == 4
    assert "0 violations" in str(report)


def test_peeking_predictor_is_caught(toy_problem):
    report = truncation_invariance(toy_problem, Peeks, k=4)
    assert not report.ok and len(report.violations) == 4
    with pytest.raises(AssertionError, match="violations"):
        assert_truncation_invariant(toy_problem, Peeks, k=2)


def test_full_sample_statistic_is_caught(toy_problem):
    report = truncation_invariance(toy_problem, Remembers, k=3)
    assert not report.ok


def test_stateful_agent_passes():
    problem = make_problem()
    report = assert_truncation_invariant(
        problem, lambda: Agent(YesterdayPrice(), ThresholdStorage(k=2)), k=4)
    assert report.ok


def test_make_env_that_ignores_the_problem_dataset_is_refused(toy_problem):
    from dataclasses import replace

    dataset = toy_problem.load_dataset()
    original_make_env = toy_problem.make_env

    def frozen_make_env(problem, split):
        return original_make_env(replace(problem, dataset=dataset, _dataset_cache=None), split)

    sneaky = replace(toy_problem, make_env=frozen_make_env, _dataset_cache=None)
    with pytest.raises(RuntimeError, match="problem.load_dataset"):
        truncation_invariance(sneaky, Clean, k=2)
