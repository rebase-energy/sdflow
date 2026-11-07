"""Settlement: how a decision, once its window is knowable, becomes a score.

The environment queues every action until the actual fields a settlement needs
have become knowable (per each field's availability rule), then hands the
settlement a :class:`SettlementContext`: the origin, the action, the knowable
actuals over the window, and the state before and after the simulator's
transition. The settlement returns the contribution ``C(S_t, a_t, W_{t+1})``.

Two settlements ship here. :class:`ScoreSettlement` scores a forecast against
the target actuals with the problem's objective; it is what makes a
:class:`~sdflow.envs.forecast.ForecastEnv`. :class:`MarketSettlement` settles
a committed quantity at a two-price market (HEFTCom24's rule). A storage
arbitrage, an inventory order, a reservoir release each get their own
settlement by implementing :meth:`Settlement.settle`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import typing as t

import pandas as pd

from ..problems.metrics import TradingRevenue
from ..problems.objective import Objective
from ..problems.schedule import Origin


@dataclass
class SettlementContext:
    """Everything a settlement may look at for one origin."""

    origin: Origin
    action: pd.DataFrame
    actuals: t.Mapping[str, pd.DataFrame]   # field -> rows knowable over the window
    state_before: t.Any = None
    state_after: t.Any = None

    def series(self, field: str, column: t.Optional[str] = None) -> pd.Series:
        """One column of an actual field, reindexed to the origin's targets.
        ``column`` defaults to the field's only value column."""
        frame = self.actuals[field]
        if column is None:
            if frame.shape[1] != 1:
                raise ValueError(
                    f"field {field!r} has columns {list(frame.columns)}; name the column"
                )
            column = frame.columns[0]
        return frame[column].reindex(self.origin.target_index)


class Settlement(ABC):
    """Turns a settled decision into a score.

    ``fields`` names the actual fields the settlement reads; the environment
    waits until every one of them is knowable over the decision window before
    calling :meth:`settle`, so a settlement can never see data a live
    participant would not have had.
    """

    fields: t.Tuple[str, ...] = ()

    @abstractmethod
    def settle(self, ctx: SettlementContext) -> float:
        """The contribution of this decision (sign as the objective defines)."""

    def target(self, ctx: SettlementContext) -> t.Optional[pd.Series]:
        """The actual series the decision was scored against, if there is one.
        Feeds ``n_scored`` and the analyzers; ``None`` for pure decision problems."""
        return None


class ScoreSettlement(Settlement):
    """``objective.calculate(actuals, prediction)``: the forecasting case."""

    def __init__(self, objective: Objective, target_field: str,
                 target_column: t.Optional[str] = None):
        self.objective = objective
        self.target_field = target_field
        self.target_column = target_column
        self.fields = (target_field,)

    def target(self, ctx: SettlementContext) -> pd.Series:
        return ctx.series(self.target_field, ctx.origin.column or self.target_column)

    def settle(self, ctx: SettlementContext) -> float:
        return self.objective.calculate(self.target(ctx), ctx.action)


class MarketSettlement(Settlement):
    """Two-price settlement of a committed quantity against the realised one:

        quantity × DA_price + (actual − quantity) × SS_price − λ·(actual − quantity)²

    HEFTCom24's rule, computed by :class:`~sdflow.problems.metrics.TradingRevenue`.
    Needs the target field (what was actually delivered) and a prices field
    with the day-ahead and system-price columns; settles only once both are
    knowable, which is when a real trader learns how the day went.
    """

    def __init__(self, revenue: TradingRevenue, target_field: str, *,
                 prices_field: str = "prices", da_column: str = "da_price",
                 ss_column: str = "ss_price", quantity_column: str = "bid",
                 target_column: t.Optional[str] = None):
        self.revenue = revenue
        self.target_field = target_field
        self.target_column = target_column
        self.prices_field = prices_field
        self.da_column = da_column
        self.ss_column = ss_column
        self.quantity_column = quantity_column
        self.fields = (target_field, prices_field)

    def target(self, ctx: SettlementContext) -> pd.Series:
        return ctx.series(self.target_field, ctx.origin.column or self.target_column)

    def settle(self, ctx: SettlementContext) -> float:
        prices = ctx.actuals[self.prices_field].reindex(ctx.origin.target_index)
        revenue = self.revenue.revenue(
            self.target(ctx), ctx.action[self.quantity_column],
            prices[self.da_column], prices[self.ss_column],
        )
        return float(revenue.sum())
