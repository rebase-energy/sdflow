"""Baselines for the signal game: the informed reader, the learner, and chance.

``BayesDig`` knows the channel model and reads every sensor: the ceiling an
agent can reach without knowing the place. ``LearnedDig`` knows nothing about
the sensors and learns the channel model from revealed places, first from the
training view in ``fit`` and then online through ``learn`` as episodes settle.
``ChanceDig`` digs at the prior's best place and ignores the sensors.
"""

from __future__ import annotations

import typing as t

import numpy as np
import pandas as pd

from sdflow.models import Agent, decision

from .problem import DIG_COL, HIDDEN_FIELD, SENSORS_FIELD, SignalTask, sensor_columns


def _posterior(likelihood: np.ndarray, prior: np.ndarray, readings: np.ndarray) -> np.ndarray:
    post = prior.copy()
    for k, o in enumerate(readings):
        post = post * likelihood[k, :, int(o)]
    s = post.sum()
    return post / s if s > 0 else prior


def _dig(post: np.ndarray, payoff: np.ndarray) -> int:
    return int(np.argmax(post @ payoff))


def _latest_readings(obs, columns: t.Sequence[str]) -> np.ndarray:
    row = obs.history(SENSORS_FIELD, window="1h")[list(columns)].iloc[-1]
    return row.to_numpy()


class BayesDig(Agent):
    """Knows ``task.likelihood``; digs where the posterior expected payoff is largest."""

    def __init__(self, task: SignalTask, name: t.Optional[str] = None):
        super().__init__(name=name or "bayes")
        self.task = task
        self.columns = sensor_columns(task)

    def act(self, obs) -> pd.DataFrame:
        post = _posterior(self.task.likelihood, self.task.prior, _latest_readings(obs, self.columns))
        return decision(obs.target_index, **{DIG_COL: _dig(post, self.task.payoff)})


class LearnedDig(Agent):
    """Tabular learner: one Dirichlet(α) per (sensor, place) cell, posterior-mean
    likelihood. Fit on the training view, then updated online from every settled
    episode (the place is revealed when the next origin opens)."""

    def __init__(self, n_places: int, n_sensors: int, n_symbols: int, *, alpha: float = 1.0,
                 name: t.Optional[str] = None):
        super().__init__(name=name or "learner")
        self.n_places, self.n_sensors, self.n_symbols = n_places, n_sensors, n_symbols
        self.alpha = float(alpha)
        self.counts = np.full((n_sensors, n_places, n_symbols), self.alpha)
        self.payoff = np.eye(n_places)
        self.columns = [f"sensor_{k}" for k in range(n_sensors)]
        self._seen: t.Dict[pd.Timestamp, np.ndarray] = {}

    @property
    def likelihood(self) -> np.ndarray:
        return self.counts / self.counts.sum(axis=2, keepdims=True)

    @property
    def prior(self) -> np.ndarray:
        return np.full(self.n_places, 1.0 / self.n_places)

    def _update(self, readings: np.ndarray, place: int) -> None:
        for k, o in enumerate(readings):
            self.counts[k, place, int(o)] += 1.0

    def fit(self, train) -> "LearnedDig":
        """Every training episode whose place is knowable: sensors at ``T - 1h``,
        place at ``T``."""
        sensors = train.history(SENSORS_FIELD)[self.columns]
        hidden = train.history(HIDDEN_FIELD)["place"]
        aligned = sensors.copy()
        aligned.index = aligned.index + pd.Timedelta("1h")
        joined = aligned.join(hidden, how="inner")
        for row in joined.itertuples(index=False):
            self._update(np.asarray(row[:-1]), int(row[-1]))
        return self

    def act(self, obs) -> pd.DataFrame:
        readings = _latest_readings(obs, self.columns)
        self._seen[obs.asof] = readings
        post = _posterior(self.likelihood, self.prior, readings)
        return decision(obs.target_index, **{DIG_COL: _dig(post, self.payoff)})

    def learn(self, record) -> None:
        readings = self._seen.pop(record.origin.asof, None)
        if readings is None or record.actuals is None or record.actuals.isna().all():
            return
        self._update(readings, int(record.actuals.iloc[0]))


class ChanceDig(Agent):
    """Ignores the sensors; digs at the prior's best place."""

    def __init__(self, n_places: int, name: t.Optional[str] = None):
        super().__init__(name=name or "chance")
        self.place = int(np.argmax(np.full(n_places, 1.0 / n_places) @ np.eye(n_places)))

    def act(self, obs) -> pd.DataFrame:
        return decision(obs.target_index, **{DIG_COL: self.place})
