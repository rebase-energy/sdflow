"""Live mode: the same loop, one origin per call, against the clock.

In a backtest the environment advances itself through the origin list. Live,
the world advances: a process wakes up when an origin opens (a scheduler, a
cron job, the toolkit's runner), is handed the data knowable *now*, and must
settle what has become knowable, decide for the open origin, and remember its
pending decisions and state until the next wake-up. :class:`LiveSession` is
that memory plus the four calls of the loop, built on the same
:class:`~sdflow.envs.base.DecisionEnv` pieces a backtest uses.

The contract that makes live and backtest one engine is replay: driving a
session with the dataset truncated at each origin reproduces the backtest's
decisions and scores exactly. The data source is the adapter: a backtest
replays history, a live runner fetches from the platform and builds a
:class:`~sdflow.data.dataset.Dataset` as of now. sdflow does not fetch; it
defines what the runner must hand it and what it gets back.
"""

from __future__ import annotations

from dataclasses import dataclass, field as _dc_field, replace
import datetime as _dt
import typing as t

import pandas as pd

from ..data import Dataset
from ..problems.problem import Problem, Splits
from ..problems.schedule import IssueSchedule, Origin

if t.TYPE_CHECKING:  # pragma: no cover
    from ..envs.base import DecisionEnv, SettlementRecord

__all__ = ["Clock", "WallClock", "FixedClock", "LiveSession", "LiveStep"]


class Clock(t.Protocol):
    def now(self) -> pd.Timestamp: ...


class WallClock:
    """The real clock, in UTC."""

    def now(self) -> pd.Timestamp:
        return pd.Timestamp(_dt.datetime.now(_dt.timezone.utc))


class FixedClock:
    """A clock that reads whatever it was set to; tests and replays set it."""

    def __init__(self, now):
        self._now = pd.Timestamp(now)

    def now(self) -> pd.Timestamp:
        return self._now

    def set(self, now) -> None:
        self._now = pd.Timestamp(now)


@dataclass
class LiveStep:
    """What one wake-up did."""

    now: pd.Timestamp
    origin: t.Optional[Origin]
    action: t.Optional[pd.DataFrame]
    settled: t.List["SettlementRecord"]
    reward: float


@dataclass
class LiveSession:
    """A policy running one origin at a time.

    Parameters
    ----------
    problem:
        The problem definition: schedule, training cutoff, objective, env factory.
    model:
        The agent (``bind``/``fit``/``act``/``learn`` contract). Fit once, on
        the first call, from the training view the environment serves; after
        that it only learns from settlements, as it would in production.
    clock:
        Where "now" comes from when :meth:`step` is not told. Default: UTC wall clock.

    Memory between calls: ``state``, ``pending``, ``last_origin`` and
    ``started``. They are plain Python and pandas objects; persist them and
    hand them back through :meth:`restore` when the process does not live from
    one wake-up to the next.
    """

    problem: Problem
    model: t.Any
    clock: Clock = _dc_field(default_factory=WallClock)
    state: t.Any = None
    pending: t.Tuple[t.Tuple[Origin, pd.DataFrame], ...] = ()
    last_origin: t.Optional[pd.Timestamp] = None
    started: bool = False

    def restore(self, *, state, pending, last_origin=None, started: bool = True) -> "LiveSession":
        self.state = state
        self.pending = tuple(pending)
        self.last_origin = pd.Timestamp(last_origin) if last_origin is not None else None
        self.started = started
        return self

    # -- what is due ----------------------------------------------------------------

    def due_origin(self, now) -> t.Optional[Origin]:
        """The latest origin that has opened by ``now`` and has not been decided."""
        now = pd.Timestamp(now)
        start = self.problem.splits.train_end
        candidates = [o for o in self.problem.schedule.origins(start, now + pd.Timedelta(days=366))
                      if o.asof <= now and (self.last_origin is None or o.asof > self.last_origin)]
        return candidates[-1] if candidates else None

    # -- one wake-up -------------------------------------------------------------------

    def _env(self, dataset: Dataset, origins: t.Sequence[Origin], now: pd.Timestamp) -> "DecisionEnv":
        """The problem's own environment over ``dataset`` and exactly ``origins``.
        Splits are widened so the problem's ``make_env`` (which asks for the
        validation split's origins) gets the live ones whatever the date."""
        train_end = self.problem.splits.train_end
        far = max(now, train_end) + pd.Timedelta(days=3660)
        live = replace(
            self.problem, dataset=dataset, schedule=IssueSchedule.explicit(list(origins)),
            splits=Splits(train_end=train_end, validation=(train_end, far),
                          holdout=(far, far + pd.Timedelta(days=1))),
            _dataset_cache=None,
        )
        return live.make_env(live, "validation")

    def step(self, dataset: Dataset, now=None) -> LiveStep:
        """One wake-up: settle what ``dataset`` makes knowable at ``now``, then
        decide for the origin open now (if any).

        ``dataset`` is everything the runner could fetch as of ``now``; the feed
        still applies each field's availability rule on top, so handing it more
        than is knowable is safe, and handing it less only delays settlement.
        """
        now = pd.Timestamp(now) if now is not None else self.clock.now()
        origin = self.due_origin(now)
        anchor = [origin] if origin is not None else [p[0] for p in self.pending]
        env = self._env(dataset, anchor or [Origin(now, pd.DatetimeIndex([now]))], now)
        env.reset()

        if not self.started:
            if hasattr(self.model, "bind"):
                self.model.bind(self.problem)
            if env.train_end is not None and hasattr(self.model, "fit"):
                self.model.fit(env.feed.view(env.train_end, strict=True))
            self.state = env.state          # the simulator's initial state
            self.started = True
        env.restore(self.state, self.pending)

        settled = env.settle(now)
        learn = getattr(self.model, "learn", None)
        if learn is not None:
            for record in settled:
                learn(record)

        action = None
        if origin is not None:
            obs = env.observe(origin)
            act = self.model.act if hasattr(self.model, "act") else self.model.predict
            action = env.commit(origin, act(obs))
            self.last_origin = origin.asof

        self.state, self.pending = env.state, env.pending
        return LiveStep(now=now, origin=origin, action=action, settled=settled,
                        reward=env.reward(settled))
