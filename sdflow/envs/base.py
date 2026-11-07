"""DecisionEnv: the sequential decision loop as a gymnasium environment.

At each :class:`~sdflow.problems.schedule.Origin` the agent sees an
:class:`Observation`: the data knowable at that clock time (a
:class:`~sdflow.data.feed.TimeView`), the target times to decide for, the
current **state** and the decisions still **pending** settlement. It returns an
action, a DataFrame on the target index validated by the env's
:class:`~sdflow.envs.actions.ActionSpec`.

Rewards are delayed to settlement, as in reality. Once a decision's window has
become knowable for every field the settlement and simulator need, the env
advances the state through the :class:`~sdflow.models.simulator.Simulator`
(``S_{t+1} = S^M(S_t, a_t, W_{t+1})``) and scores the decision with the
:class:`~sdflow.envs.settlement.Settlement` (``C(S_t, a_t, W_{t+1})``). Without
a simulator the state is ``None`` and the env is a pure scoring loop, which is
what :class:`~sdflow.envs.forecast.ForecastEnv` is.
"""

from __future__ import annotations

from dataclasses import dataclass
import typing as t

import gymnasium as gym
import pandas as pd

from ..data import DataFeed, TimeView
from ..problems.objective import Objective
from ..problems.schedule import Origin
from .actions import ActionSpec, ColumnsAction
from .settlement import Settlement, SettlementContext

if t.TYPE_CHECKING:  # pragma: no cover
    from ..models.simulator import Simulator


class Observation(TimeView):
    """What an agent sees at one origin: point-in-time data access (via
    :class:`TimeView`), the target index to decide for, the current state and
    the decisions pending settlement. ``column`` names the target-field column
    this origin is scoped to (multi-zone problems); None means the env's default."""

    def __init__(self, feed: DataFeed, origin: Origin, state: t.Any = None,
                 pending: t.Sequence[t.Tuple[Origin, pd.DataFrame]] = ()):
        super().__init__(feed, origin.asof)
        self.target_index = origin.target_index
        self.column = origin.column
        self.state = state
        self.pending = tuple(pending)

    def __repr__(self):
        extra = f", state={self.state!r}" if self.state is not None else ""
        return (f"Observation(asof={self.asof}, targets={self.target_index[0]} → "
                f"{self.target_index[-1]}, n={len(self.target_index)}{extra})")


@dataclass
class SettlementRecord:
    """One origin settled: its action, the actuals it was scored against (if
    any), the score, and the state after its transition."""

    origin: Origin
    action: pd.DataFrame
    actuals: t.Optional[pd.Series]
    score: float
    state: t.Any = None

    @property
    def prediction(self) -> pd.DataFrame:
        """The action, under the name forecasting code uses."""
        return self.action

    @property
    def n_scored(self) -> int:
        """Target rows the score covers: non-missing actuals, or every row of the
        action when the settlement has no target series."""
        if self.actuals is None:
            return len(self.action)
        return int(self.actuals.notna().sum())


class DecisionEnv(gym.Env):
    """Sequential decisions over a fixed list of origins, with delayed settlement.

    Parameters
    ----------
    feed:
        The point-in-time data gate.
    origins:
        The origins to step through (from ``problem.origins(split)``).
    settlement:
        Scores each decision once its window is knowable.
    objective:
        The problem's objective; its direction turns scores into rewards
        (more reward is always better).
    action_spec:
        Validates submitted actions. Default: any DataFrame on the target index.
    simulator:
        Transition model carrying state between origins. None for stateless
        problems.
    train_end:
        Clock time of the training view served in ``reset()``'s ``info["train"]``.
    target_field, target_column:
        The actual field (and column) decisions are about, when there is one;
        analyzers use them for baselines.
    """

    metadata = {"render_modes": []}

    def __init__(self, feed: DataFeed, origins: t.Sequence[Origin],
                 settlement: Settlement, objective: Objective, *,
                 action_spec: t.Optional[ActionSpec] = None,
                 simulator: t.Optional["Simulator"] = None,
                 train_end=None, target_field: t.Optional[str] = None,
                 target_column: t.Optional[str] = None):
        if not origins:
            raise ValueError(f"{type(self).__name__} needs at least one origin")
        self.feed = feed
        self.origins = list(origins)
        self.settlement = settlement
        self.objective = objective
        self.action_spec = action_spec or ColumnsAction()
        self.simulator = simulator
        self.train_end = pd.Timestamp(train_end) if train_end is not None else None
        self.target_field = target_field
        self.target_column = target_column

        # Everything a settlement or transition reads must be knowable before
        # the origin settles: wait for the slowest field.
        needed = list(dict.fromkeys([*settlement.fields, *getattr(simulator, "fields", ())]))
        self._fields = tuple(needed)
        lags = [feed.dataset.field(name).settlement_lag for name in needed]
        self._settle_lag = max(lags) if lags else pd.Timedelta(0)

        self.state: t.Any = None
        self._i = 0
        self._pending: t.List[t.Tuple[Origin, pd.DataFrame]] = []

    @property
    def stateful(self) -> bool:
        """True when a simulator carries state between origins. Precomputed
        (vectorized) actions are only valid for stateless envs."""
        return self.simulator is not None

    # -- gym API ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._i = 0
        self._pending = []
        self.state = self.simulator.reset(seed=seed) if self.simulator is not None else None
        info = {}
        if self.train_end is not None:
            # Strict boundary: a value stamped exactly train_end is a scored
            # target of the first evaluation origin, never training data.
            info["train"] = self.feed.view(self.train_end, strict=True)
        return self.observe(self.origins[0]), info

    def step(self, action):
        origin = self.origins[self._i]
        self.commit(origin, action)
        self._i += 1

        terminated = self._i >= len(self.origins)
        settle_asof = self.flush_time() if terminated else self.origins[self._i].asof
        settled = self.settle(settle_asof)
        obs = None if terminated else self.observe(self.origins[self._i])
        return obs, self.reward(settled), terminated, False, {"settled": settled}

    # -- the loop's pieces, for runners that drive one origin at a time ---------------
    #
    # ``step`` is these four in order. A live runner calls them itself: settle
    # what has become knowable, observe the origin that is open now, commit the
    # agent's action, and persist ``pending`` and ``state`` until the next call.

    @property
    def pending(self) -> t.Tuple[t.Tuple[Origin, pd.DataFrame], ...]:
        """Decisions committed and not yet settled."""
        return tuple(self._pending)

    def restore(self, state: t.Any, pending: t.Sequence[t.Tuple[Origin, pd.DataFrame]]) -> None:
        """Resume from persisted ``state`` and ``pending`` (a live runner's memory)."""
        self.state = state
        self._pending = list(pending)

    def observe(self, origin: Origin) -> Observation:
        """What the agent sees at ``origin``: data knowable then, state, pending."""
        return Observation(self.feed, origin, state=self.state, pending=self._pending)

    def commit(self, origin: Origin, action) -> pd.DataFrame:
        """Validate ``action`` for ``origin`` and queue it for settlement."""
        action = self.action_spec.validate(action, origin)
        self._pending.append((origin, action))
        return action

    def flush_time(self) -> pd.Timestamp:
        """The clock time at which every pending decision has become knowable."""
        if not self._pending:
            return self.origins[-1].asof
        return max(o.target_end for o, _ in self._pending) + self._settle_lag

    def reward(self, settled: t.Sequence[SettlementRecord]) -> float:
        """Sum of settled scores, signed so that more reward is always better."""
        return float(sum(-r.score if self.objective.lower_is_better else r.score
                         for r in settled if pd.notna(r.score)))

    def settle(self, asof) -> t.List[SettlementRecord]:
        """Settle every pending origin whose window is knowable at ``asof``.

        Stateful envs settle strictly in origin order, since each transition
        feeds the next; stateless ones settle whatever is knowable."""
        settled, still_pending = [], []
        for k, (origin, action) in enumerate(self._pending):
            if origin.target_end + self._settle_lag > asof:
                if self.stateful:
                    still_pending.extend(self._pending[k:])
                    break
                still_pending.append((origin, action))
                continue
            actuals = {
                name: self.feed.actuals_between(name, origin.target_start,
                                                origin.target_end, asof=asof)
                for name in self._fields
            }
            before = self.state
            if self.simulator is not None:
                self.state = self.simulator.step(before, action, actuals)
            ctx = SettlementContext(origin, action, actuals, before, self.state)
            score = self.settlement.settle(ctx)
            settled.append(SettlementRecord(origin, action, self.settlement.target(ctx),
                                            score, state=self.state))
        self._pending = still_pending
        return settled
