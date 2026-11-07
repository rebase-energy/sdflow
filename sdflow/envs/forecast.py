"""ForecastEnv: rolling-origin forecasting as a real gymnasium environment.

The stateless case of :class:`~sdflow.envs.base.DecisionEnv`: the action *is*
the forecast, a DataFrame indexed by the current origin's target times with a
``"point"`` column or quantile-level float columns, and the settlement scores
it against the target actuals with the problem's objective
(:class:`~sdflow.envs.settlement.ScoreSettlement`). The model never touches the
dataset; the observation is its whole world.

Rewards are **delayed to settlement**, as in reality: submitting a day-ahead
forecast tells you nothing — the score arrives once the actuals do. At each
step the env settles every previously submitted origin whose targets have
become knowable (per the target field's availability rule) and pays their
scores as the reward (negated when the objective is lower-is-better, so more
reward is always better). ``info["settled"]`` carries the full
:class:`SettlementRecord` s for analyzers.
"""

from __future__ import annotations

import typing as t

from ..data import DataFeed
from ..problems.objective import Objective
from ..problems.schedule import Origin
from .actions import ActionSpec, ForecastAction
from .base import DecisionEnv, Observation, SettlementRecord
from .settlement import ScoreSettlement, Settlement

if t.TYPE_CHECKING:  # pragma: no cover
    from ..models.simulator import Simulator

__all__ = ["ForecastEnv", "Observation", "SettlementRecord"]


class ForecastEnv(DecisionEnv):
    """Rolling-origin forecast evaluation over a fixed list of origins.

    Parameters
    ----------
    feed:
        The point-in-time data gate.
    origins:
        The origins to step through (from ``problem.origins(split)``).
    target_field:
        Name of the actual field being forecast.
    objective:
        Scores each settled origin (prediction vs actuals).
    target_column:
        Column of the target field to score (default: its only column).
    train_end:
        Clock time of the training view served in ``reset()``'s
        ``info["train"]``.
    quantiles:
        If set, actions must provide exactly these quantile columns;
        if None, actions must provide a ``"point"`` column.
    settlement, action_spec, simulator:
        Override the scoring rule, the action contract, or add state. Defaults
        give the plain forecasting loop.
    """

    def __init__(self, feed: DataFeed, origins: t.Sequence[Origin],
                 target_field: str, objective: Objective, *,
                 target_column=None, train_end=None,
                 quantiles: t.Optional[t.Sequence[float]] = None,
                 settlement: t.Optional[Settlement] = None,
                 action_spec: t.Optional[ActionSpec] = None,
                 simulator: t.Optional["Simulator"] = None):
        if not origins:
            raise ValueError("ForecastEnv needs at least one origin")
        self.quantiles = [float(q) for q in quantiles] if quantiles is not None else None

        field = feed.dataset.field(target_field)
        if target_column is None and not all(o.column for o in origins):
            value_cols = [c for c in field.frame.columns if c != field.knowledge_col]
            if len(value_cols) != 1:
                raise ValueError(
                    f"target field {target_field!r} has columns {value_cols}; "
                    f"pass target_column or give every origin a column"
                )
            target_column = value_cols[0]
        self._target_lag = field.settlement_lag

        super().__init__(
            feed, origins,
            settlement or self._default_settlement(objective, target_field, target_column),
            objective,
            action_spec=action_spec or self._default_action_spec(),
            simulator=simulator, train_end=train_end,
            target_field=target_field, target_column=target_column,
        )

    def _default_settlement(self, objective: Objective, target_field: str,
                            target_column) -> Settlement:
        return ScoreSettlement(objective, target_field, target_column)

    def _default_action_spec(self) -> ActionSpec:
        return ForecastAction(self.quantiles)
