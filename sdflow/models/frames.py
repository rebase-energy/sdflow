"""The two frames that flow through an agent: Forecast and Decision.

Both are plain DataFrames indexed by the origin's target times; what differs
is the contract on their columns, and that contract is what lets a predictor,
an optimizer and an environment be swapped independently (LEAN's Insight →
PortfolioTarget → Order, with two stages instead of three):

* a **Forecast** has a ``"point"`` column, or float columns at quantile levels
  (``0.1, 0.5, 0.9``) — what a :class:`~sdflow.models.predictor.Predictor` emits;
* a **Decision** has named decision columns (``"bid"``, ``"flow"``) — what an
  :class:`~sdflow.models.optimizer.Optimizer` emits and an environment's
  :class:`~sdflow.envs.actions.ActionSpec` validates.

A forecast problem's decision *is* its forecast, so the aliases are the same
type; the builders below make each contract explicit at the point of creation.
"""

from __future__ import annotations

import typing as t

import numpy as np
import pandas as pd

from ..problems.metrics import POINT_COL, quantile_columns

Forecast = pd.DataFrame
Decision = pd.DataFrame


def forecast(index: pd.DatetimeIndex, point=None,
             quantiles: t.Optional[t.Mapping[float, t.Any]] = None) -> Forecast:
    """A Forecast on ``index``: a point forecast, quantile forecasts, or both."""
    if point is None and not quantiles:
        raise ValueError("a forecast needs a point value or quantile values")
    data: t.Dict[t.Any, t.Any] = {}
    if point is not None:
        data[POINT_COL] = point
    for level, values in (quantiles or {}).items():
        data[float(level)] = values
    return pd.DataFrame(data, index=index)


def decision(index: pd.DatetimeIndex, **columns) -> Decision:
    """A Decision on ``index`` with the named decision columns."""
    if not columns:
        raise ValueError("a decision needs at least one named column")
    return pd.DataFrame(columns, index=index)


def quantile_levels(fc: Forecast) -> t.List[float]:
    """The quantile levels a forecast carries (empty for a point forecast)."""
    return quantile_columns(fc)


def point_of(fc: Forecast) -> pd.Series:
    """The point forecast: the ``"point"`` column, else the median quantile, else
    the quantile nearest 0.5, else the single column."""
    if POINT_COL in fc.columns:
        return fc[POINT_COL]
    levels = quantile_columns(fc)
    if levels:
        nearest = min(levels, key=lambda q: abs(q - 0.5))
        return fc[nearest]
    if fc.shape[1] == 1:
        return fc.iloc[:, 0]
    raise ValueError(f"cannot pick a point forecast from columns {list(fc.columns)}")


def quantile_of(fc: Forecast, level: float) -> pd.Series:
    """The forecast at quantile ``level``, interpolating linearly between the
    levels the forecast carries (a point forecast is returned for any level)."""
    levels = quantile_columns(fc)
    if not levels:
        return point_of(fc)
    level = float(level)
    if level in levels:
        return fc[level]
    lo = max((q for q in levels if q < level), default=None)
    hi = min((q for q in levels if q > level), default=None)
    if lo is None or hi is None:
        return fc[min(levels, key=lambda q: abs(q - level))]
    w = (level - lo) / (hi - lo)
    return pd.Series(np.asarray(fc[lo]) * (1 - w) + np.asarray(fc[hi]) * w,
                     index=fc.index, name=level)
