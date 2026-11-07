"""Reference baselines for GEFCom2014: the competition's own naive benchmarks.

The official leaderboards carried one "Benchmark" entry per track, generated
by the organizers "using naive models" (Hong et al., IJF 2016) without a
published recipe. :class:`OfficialBenchmark` reproduces them per track,
identified by matching holdout (tasks 4-15) mean pinball against the
published benchmark scores:

* ``wind`` — unconditional climatology: empirical 1..99% quantiles of all
  past power observations. Scores 0.08685 vs the official 0.08671 (+0.2%).
* ``solar`` — deterministic hour-of-day climatology mean, repeated across
  all 99 quantiles. Scores 0.03798 vs the official 0.03759 (+1.0%).
* ``load``/``price`` — not yet reproduced; the hour-of-day climatology
  quantiles of :class:`ClimatologyQuantiles` stand in.

:class:`ClimatologyQuantiles` remains available as the stronger generic
baseline: empirical quantiles by hour of day over a trailing window. Beating
either requires actually using the provided explanatory variables
(temperature stations, load forecasts, ECMWF fields), which is where the
historical field earned their ranks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sdflow.models.predictor import Predictor

from .problem import QUANTILES, TRACKS


class ClimatologyQuantiles(Predictor):
    output_kind = "quantiles"
    quantiles = QUANTILES

    def __init__(self, window="120D", quantiles=None, target_field=None,
                 name="climatology-quantiles"):
        super().__init__(name=name)
        self.window = window
        if quantiles is not None:
            self.quantiles = tuple(quantiles)
        self.target_field = target_field

    def _target_field(self) -> str:
        if self.target_field is not None:
            return self.target_field
        track = self.problem.name.split(":")[1]
        return TRACKS[track]["target_field"]

    def predict(self, obs) -> pd.DataFrame:
        hist = obs.history(self._target_field(), window=self.window)
        series = hist[obs.column].dropna()
        q = np.asarray(self.quantiles)

        by_hour = {}
        fallback = (np.quantile(series.to_numpy(), q) if len(series)
                    else np.zeros(len(q)))
        if len(series):
            for hour, vals in series.groupby(series.index.hour):
                if len(vals) >= 10:
                    by_hour[hour] = np.quantile(vals.to_numpy(), q)

        rows = [by_hour.get(t.hour, fallback) for t in obs.target_index]
        return pd.DataFrame(np.vstack(rows), index=obs.target_index,
                            columns=list(self.quantiles))


class OfficialBenchmark(ClimatologyQuantiles):
    """Per-track reproduction of the official "Benchmark" leaderboard entries
    (see the module docstring for the identification); the parent's
    hour-of-day climatology stands in for the load and price tracks."""

    def __init__(self, name="gefcom2014-benchmark"):
        super().__init__(name=name)

    def predict(self, obs) -> pd.DataFrame:
        track = self.problem.name.split(":")[1]
        if track not in ("wind", "solar"):
            return super().predict(obs)
        hist = obs.history(self._target_field())  # all available history
        series = hist[obs.column].dropna()
        q = np.asarray(self.quantiles)
        if track == "wind":
            row = np.quantile(series.to_numpy(), q) if len(series) else np.zeros(len(q))
            data = np.tile(row, (len(obs.target_index), 1))
        else:
            overall = float(series.mean()) if len(series) else 0.0
            by_hour = series.groupby(series.index.hour).mean().to_dict()
            points = np.asarray([by_hour.get(t.hour, overall) for t in obs.target_index])
            data = np.tile(points[:, None], (1, len(q)))
        return pd.DataFrame(data, index=obs.target_index, columns=list(self.quantiles))


def get_model() -> OfficialBenchmark:
    """Submission-convention factory: the competition's own benchmark, so a
    search's baseline candidate starts where the official leaderboard's
    "Benchmark" entry sits."""
    return OfficialBenchmark()
