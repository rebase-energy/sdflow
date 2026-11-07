"""TradingEnv: day-ahead commitment on top of the rolling-origin loop.

The action carries a ``"bid"`` column: the quantity committed at the origin
for each settlement period. Settlement is :class:`~sdflow.envs.settlement.MarketSettlement`
with HEFTCom24's two-price rule, each period earning

    bid × DA_price + (actual − bid) × SS_price − 0.07·(actual − bid)²

(verified against the official per-period revenues in the competition
archive). Reward is the (positive) revenue, credited only once both the
delivered actuals *and* the prices are knowable — you find out how a trade
went when the settlement data lands, exactly as in reality.
"""

from __future__ import annotations

from ..problems.metrics import TradingRevenue
from ..problems.objective import Objective
from .actions import ActionSpec, ColumnsAction
from .forecast import ForecastEnv
from .settlement import MarketSettlement, Settlement

BID_COL = "bid"


class TradingEnv(ForecastEnv):
    """ForecastEnv + market settlement.

    Extra parameters: ``prices_field`` (an actual field with day-ahead and
    system-price columns) and the two column names.
    """

    def __init__(self, *args, prices_field: str = "prices",
                 da_column: str = "da_price", ss_column: str = "ss_price",
                 **kwargs):
        kwargs.setdefault("quantiles", None)
        self.prices_field = prices_field
        self.da_column = da_column
        self.ss_column = ss_column
        super().__init__(*args, **kwargs)

    def _default_settlement(self, objective: Objective, target_field: str,
                            target_column) -> Settlement:
        metric = objective.metric
        if not isinstance(metric, TradingRevenue):
            raise TypeError("TradingEnv requires a TradingRevenue objective")
        return MarketSettlement(
            metric, target_field, prices_field=self.prices_field,
            da_column=self.da_column, ss_column=self.ss_column,
            quantity_column=BID_COL, target_column=target_column,
        )

    def _default_action_spec(self) -> ActionSpec:
        return ColumnsAction(BID_COL, what="bids")
