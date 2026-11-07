"""Optimizer: the decision stage of a policy.

Given the forecast the predictor produced and the observation (state, pending
decisions, point-in-time data), an optimizer returns the :data:`Decision` the
environment will settle. It is where a problem's economics live: the critical
fractile of a newsvendor, the schedule of a store against a price forecast, a
bid shaped by the imbalance penalty. The same forecast can feed any of them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import typing as t

from .frames import Decision, Forecast, point_of, quantile_of
from .model import Model


class Optimizer(Model, ABC):
    def __init__(self, name: t.Optional[str] = None):
        super().__init__(name or type(self).__name__)

    def bind(self, problem) -> "Optimizer":
        """Called before a run with the Problem; default stores it."""
        self.problem = problem
        return self

    @abstractmethod
    def decide(self, forecast: t.Optional[Forecast], obs) -> Decision:
        """The decision for ``obs.target_index`` given ``forecast`` (None when the
        agent has no predictor) and the observation's state and data."""


class IdentityOptimizer(Optimizer):
    """The forecast is the decision: a forecast problem expressed as an agent."""

    def decide(self, forecast, obs) -> Decision:
        if forecast is None:
            raise ValueError("IdentityOptimizer needs a forecast to pass through")
        return forecast


class QuantileDecision(Optimizer):
    """Commit the forecast at one quantile level as the decision quantity.

    The newsvendor rule: with unit cost ``c_under`` for falling short and
    ``c_over`` for overshooting, the optimal commitment is the forecast's
    ``c_under / (c_under + c_over)`` quantile. Pass ``level`` directly, or the
    two costs. A point forecast is committed as is.
    """

    def __init__(self, level: t.Optional[float] = None, *, c_under: t.Optional[float] = None,
                 c_over: t.Optional[float] = None, column: str = "bid",
                 name: t.Optional[str] = None):
        super().__init__(name)
        if level is None:
            if c_under is None or c_over is None:
                raise ValueError("pass level, or both c_under and c_over")
            if c_under < 0 or c_over < 0 or c_under + c_over == 0:
                raise ValueError("costs must be non-negative and not both zero")
            level = c_under / (c_under + c_over)
        if not 0 <= level <= 1:
            raise ValueError("level must lie in [0, 1]")
        self.level = float(level)
        self.column = column

    def decide(self, forecast, obs) -> Decision:
        if forecast is None:
            raise ValueError("QuantileDecision needs a forecast")
        values = quantile_of(forecast, self.level)
        return Decision({self.column: values.to_numpy()}, index=forecast.index)


__all__ = ["Optimizer", "IdentityOptimizer", "QuantileDecision", "point_of"]
