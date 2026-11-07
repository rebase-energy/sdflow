"""StorageSimulator: a bounded store, the most common piece of state.

A battery's charge, a reservoir's volume, a warehouse's stock and a cash
balance are all one model: a level that each period's flow moves up or down,
inside limits. Positive flow goes **into** the store (charging, buying,
inflow), negative flow comes **out** (discharging, selling, release). Flows
the limits cannot accommodate are clipped, and the flows actually executed are
returned on the state so a settlement can price what really happened rather
than what was asked for.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import typing as t

import pandas as pd

from .simulator import Simulator


@dataclass(frozen=True)
class StorageState:
    level: float
    # executed flows over the last settled window; a record of the transition,
    # not part of the state's identity (two states with equal levels are equal)
    realized: t.Optional[pd.Series] = field(default=None, compare=False)

    def __repr__(self):
        return f"StorageState(level={self.level:.3f})"


class StorageSimulator(Simulator):
    """``level_{t+1} = clip(level_t + η_in·inflow_t − outflow_t/η_out, 0, capacity)``.

    Parameters
    ----------
    capacity:
        Upper bound on the level.
    initial_level:
        Level at ``reset``.
    max_inflow, max_outflow:
        Per-period limits on the magnitude of positive and negative flows.
    efficiency_in, efficiency_out:
        Fraction of an inflow that reaches the store; fraction of what leaves
        the store that is delivered.
    column:
        The decision column holding the flow per target period.
    """

    def __init__(self, capacity: float, *, initial_level: float = 0.0,
                 max_inflow: float = math.inf, max_outflow: float = math.inf,
                 efficiency_in: float = 1.0, efficiency_out: float = 1.0,
                 column: str = "flow", name: t.Optional[str] = None):
        super().__init__(name)
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        if not 0 <= initial_level <= capacity:
            raise ValueError("initial_level must lie in [0, capacity]")
        if not (0 < efficiency_in <= 1 and 0 < efficiency_out <= 1):
            raise ValueError("efficiencies must lie in (0, 1]")
        self.capacity = float(capacity)
        self.initial_level = float(initial_level)
        self.max_inflow = float(max_inflow)
        self.max_outflow = float(max_outflow)
        self.efficiency_in = float(efficiency_in)
        self.efficiency_out = float(efficiency_out)
        self.column = column

    def reset(self, *, seed: t.Optional[int] = None) -> StorageState:
        return StorageState(self.initial_level)

    def step(self, state: StorageState, decision: pd.DataFrame,
             exogenous: t.Mapping[str, pd.DataFrame]) -> StorageState:
        level = state.level
        realized = []
        for flow in decision[self.column].fillna(0.0).to_numpy(dtype=float):
            if flow >= 0:
                room = (self.capacity - level) / self.efficiency_in
                flow = min(flow, self.max_inflow, room)
                level += flow * self.efficiency_in
            else:
                available = level * self.efficiency_out
                flow = max(flow, -self.max_outflow, -available)
                level += flow / self.efficiency_out
            realized.append(flow)
        level = min(max(level, 0.0), self.capacity)   # guard float drift
        return StorageState(level, pd.Series(realized, index=decision.index, name=self.column))
