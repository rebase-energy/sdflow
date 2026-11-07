"""Simulator: the transition model of a sequential decision problem.

Powell's ``S_{t+1} = S^M(S_t, a_t, W_{t+1})``: given the state, the decision
taken and the exogenous information that arrived afterwards, produce the next
state. A :class:`~sdflow.envs.base.DecisionEnv` calls :meth:`step` when a
decision settles, with the knowable actuals over its window as ``exogenous``,
and hands the resulting state to the next observation.

State is whatever the simulator returns; keep it small and immutable (a frozen
dataclass) so observations can carry it safely. ``fields`` names the actual
fields the transition reads, so the environment waits for them to be knowable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import typing as t

import pandas as pd

from .model import Model


class Simulator(Model, ABC):
    fields: t.Tuple[str, ...] = ()

    def __init__(self, name: t.Optional[str] = None):
        super().__init__(name or type(self).__name__)

    def reset(self, *, seed: t.Optional[int] = None) -> t.Any:
        """The initial state. Called by the environment's ``reset``."""
        return None

    @abstractmethod
    def step(self, state: t.Any, decision: pd.DataFrame,
             exogenous: t.Mapping[str, pd.DataFrame]) -> t.Any:
        """The next state after ``decision`` was carried out over its window
        while ``exogenous`` (actual field -> frame over the window) unfolded."""
