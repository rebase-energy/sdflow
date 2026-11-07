"""Action specs: what a valid decision looks like at one origin.

Every action in sdflow is a DataFrame indexed by the origin's target times; an
:class:`ActionSpec` says which columns it must carry and normalises it. The
environment validates every submitted action through its spec before the
action enters the settlement queue, so a malformed decision fails at the step
that produced it rather than days later when it settles.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import typing as t

import pandas as pd

from ..problems.metrics import POINT_COL, quantile_columns
from ..problems.schedule import Origin


class ActionSpec(ABC):
    """Validates the action submitted at ``origin`` and returns it reindexed to
    ``origin.target_index``."""

    what: str = "values"

    def _frame(self, action, origin: Origin) -> pd.DataFrame:
        if not isinstance(action, pd.DataFrame):
            raise TypeError(
                f"action must be a DataFrame indexed by target time, got "
                f"{type(action).__name__}"
            )
        missing = origin.target_index.difference(action.index)
        if len(missing):
            raise ValueError(
                f"action is missing {self.what} for {len(missing)} target times "
                f"(first: {missing[0]})"
            )
        return action.reindex(origin.target_index)

    @abstractmethod
    def validate(self, action, origin: Origin) -> pd.DataFrame:
        """Return the normalised action or raise ``TypeError``/``ValueError``."""


class ForecastAction(ActionSpec):
    """A forecast: a ``"point"`` column, or float columns for exactly the given
    quantile levels. A single unnamed column is accepted as the point forecast."""

    what = "forecasts"

    def __init__(self, quantiles: t.Optional[t.Sequence[float]] = None):
        self.quantiles = [float(q) for q in quantiles] if quantiles is not None else None

    def validate(self, action, origin: Origin) -> pd.DataFrame:
        action = self._frame(action, origin)
        if self.quantiles is not None:
            have, need = set(quantile_columns(action)), set(self.quantiles)
            if not need.issubset(have):
                raise ValueError(
                    f"action must provide quantile columns {sorted(need)}, "
                    f"got {sorted(have) or list(action.columns)}"
                )
        elif POINT_COL not in action.columns:
            if action.shape[1] == 1:
                action = action.set_axis([POINT_COL], axis=1)
            else:
                raise ValueError(
                    f"action must have a {POINT_COL!r} column, got {list(action.columns)}"
                )
        return action


class ColumnsAction(ActionSpec):
    """Named decision columns (``"bid"``, ``"flow"``), all required. With no
    columns any DataFrame on the target index is accepted."""

    def __init__(self, *columns: str, what: str = "decisions"):
        self.columns = tuple(columns)
        self.what = what

    def validate(self, action, origin: Origin) -> pd.DataFrame:
        if not isinstance(action, pd.DataFrame):
            raise TypeError("action must be a DataFrame indexed by target time")
        missing = [c for c in self.columns if c not in action.columns]
        if missing:
            cols = f"a {missing[0]!r} column" if len(self.columns) == 1 else f"columns {list(self.columns)}"
            raise ValueError(f"{self.what} action must have {cols}, got {list(action.columns)}")
        return self._frame(action, origin)
