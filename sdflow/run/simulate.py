"""Simulation mode: the same loop over sampled exogenous information.

A backtest runs the agent over the one history that happened. A simulation
runs it over many draws of what could happen, from the problem's ``sampler``
(``seed -> Dataset``), and reports the distribution of scores. Nothing else
changes: the environment, the settlement, the agent and the splits are the
ones the backtest uses, so a score here and a score on history are the same
measurement over different data. The signal game is the first problem with a
sampler; a weather-driven problem gets one from an ensemble or a generator.
"""

from __future__ import annotations

from dataclasses import dataclass, field as _dc_field
import typing as t

import pandas as pd

from ..problems.problem import Problem
from .experiment import Experiment
from .result import Result

__all__ = ["simulate", "SimulationResult"]


@dataclass
class SimulationResult:
    problem: str
    model: str
    results: t.List[Result] = _dc_field(default_factory=list)
    seeds: t.List[int] = _dc_field(default_factory=list)

    @property
    def scores(self) -> pd.Series:
        return pd.Series([r.score for r in self.results], index=pd.Index(self.seeds, name="seed"),
                         name="score")

    @property
    def score(self) -> float:
        """Mean score over scenarios."""
        return float(self.scores.mean())

    @property
    def spread(self) -> float:
        """Standard deviation of the score over scenarios (0 for one scenario)."""
        return float(self.scores.std(ddof=0)) if len(self.scores) > 1 else 0.0

    def quantile(self, q: float) -> float:
        return float(self.scores.quantile(q))

    def __repr__(self):
        return (f"SimulationResult({self.problem!r}, {self.model!r}, n={len(self.results)}, "
                f"score={self.score:.4f} ± {self.spread:.4f})")


def simulate(problem: Problem, make_model: t.Callable[[], t.Any], *, n_scenarios: int = 10,
             seed: int = 0, split: str = "validation", analyzers=None,
             mode: str = "auto") -> SimulationResult:
    """Run ``make_model()`` on ``n_scenarios`` draws of the problem's exogenous
    information and collect the results. Scenario ``i`` uses seed ``seed + i``."""
    if n_scenarios < 1:
        raise ValueError("n_scenarios must be >= 1")
    out = SimulationResult(problem=problem.name, model="")
    for i in range(n_scenarios):
        s = int(seed) + i
        model = make_model()
        result = Experiment(problem.scenario(s), model, analyzers=analyzers, split=split).run(mode=mode)
        out.results.append(result)
        out.seeds.append(s)
        out.model = result.model
    return out
