"""Reference baseline for fingrid:wind: a capacity-normalised power curve.

Bin the capacity-weighted mean hub-height wind speed across Finland, and
predict each bin's mean *capacity factor* over the training period, then
multiply back by the capacity known at the origin. Normalising is what makes
one curve valid across a window in which the fleet quadrupled: a raw-MW curve
fitted on 2022-2025 would systematically under-forecast 2026.

No turbine physics, no direction, no icing — just "this much wind over
Finland has historically meant this share of the fleet running". It is the
floor a weather-driven model must clear; a candidate that cannot beat it is
not using the NWP.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sdflow.features import ForecastField, Lag
from sdflow.models.predictor import FeaturePredictor

from .points import FI_POINTS
from .problem import CAPACITY_FIELD, NWP_FIELD, TARGET_FIELD

N_BINS = 40
SPEED_COLUMNS = tuple(f"{p.name}__WindSpeed_100" for p in FI_POINTS)
WEIGHTS = np.array([p.weight for p in FI_POINTS], dtype=float)
WEIGHTS = WEIGHTS / WEIGHTS.sum()


class PowerCurve(FeaturePredictor):
    features = (
        ForecastField(NWP_FIELD, columns=SPEED_COLUMNS),
        Lag(CAPACITY_FIELD, ("24h",)),
    )

    def __init__(self, name="power-curve"):
        self._speeds = None            # bin centres (m/s)
        self._capacity_factor = None   # mean CF per bin
        super().__init__(features=self.features, name=name)

    @staticmethod
    def _split(X: pd.DataFrame):
        """(zone wind speed, capacity) — speeds first, capacity lag last."""
        speeds = X.iloc[:, :len(SPEED_COLUMNS)].to_numpy(dtype=float)
        capacity = X.iloc[:, -1].to_numpy(dtype=float)
        return np.nansum(speeds * WEIGHTS, axis=1), capacity

    def fit(self, train) -> "PowerCurve":
        from sdflow.features.materialize import supervised_frame

        feed = train.feed if hasattr(train, "feed") else train
        X, y = supervised_frame(feed, self.features, self.problem.origins("train"),
                                TARGET_FIELD)
        speed, capacity = self._split(X)
        target = y.to_numpy(dtype=float)
        ok = np.isfinite(speed) & np.isfinite(target) & np.isfinite(capacity) & (capacity > 0)
        speed, factor = speed[ok], target[ok] / capacity[ok]
        if not len(speed):
            raise RuntimeError("power-curve baseline: no usable training rows")

        edges = np.unique(np.quantile(speed, np.linspace(0, 1, N_BINS + 1)))
        which = np.clip(np.digitize(speed, edges[1:-1]), 0, len(edges) - 2)
        centres, factors = [], []
        for b in range(len(edges) - 1):
            rows = which == b
            if rows.sum() >= 10:
                centres.append(speed[rows].mean())
                factors.append(factor[rows].mean())
        self._speeds = np.asarray(centres)
        self._capacity_factor = np.asarray(factors)
        return self

    def predict_tabular(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._speeds is None:
            raise RuntimeError("power-curve baseline used before fit()")
        speed, capacity = self._split(X)
        # np.interp holds the ends flat, which is the right shape anyway:
        # below cut-in and above rated, output stops tracking wind speed.
        factor = np.interp(speed, self._speeds, self._capacity_factor)
        factor = np.where(np.isfinite(speed), factor, self._capacity_factor.mean())
        capacity = np.where(np.isfinite(capacity) & (capacity > 0),
                            capacity, np.nanmax(capacity) if np.isfinite(capacity).any() else 0.0)
        return pd.DataFrame({"point": np.clip(factor * capacity, 0.0, None)},
                            index=X.index)


def get_model() -> PowerCurve:
    return PowerCurve()
