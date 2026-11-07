"""Agent: the policy, composed from a predictor and an optimizer.

Powell's taxonomy maps one to one onto the stages of a decision: the
:class:`~sdflow.models.predictor.Predictor` forecasts the exogenous
information, the :class:`~sdflow.models.optimizer.Optimizer` turns that
forecast plus the state into a :data:`~sdflow.models.frames.Decision`, and the
environment settles it. An Agent wires the stages together and is what the
:class:`~sdflow.run.experiment.Experiment` drives, through the same
``bind`` / ``fit`` / ``act`` contract a bare predictor satisfies (a predictor
on its own is the agent whose optimizer is the identity).

The same predictor can therefore be scored on a forecast leaderboard and, with
an optimizer behind it, on a revenue leaderboard, without being rewritten.
"""

from __future__ import annotations

import typing as t

import pandas as pd

from .frames import Decision, Forecast
from .model import Model
from .optimizer import Optimizer
from .predictor import Predictor

if t.TYPE_CHECKING:  # pragma: no cover
    from ..envs.base import SettlementRecord
    from .simulator import Simulator


class Agent(Model):
    """Policy π: ``act(obs)`` = ``optimizer.decide(predictor.predict(obs), obs)``.

    Parameters
    ----------
    predictor:
        Forecasts at each origin. Optional: a rule-based agent may decide
        from the observation alone.
    optimizer:
        Turns forecast and observation into the decision. Optional: without
        one the forecast is the decision.
    simulator:
        A world model the agent may use for lookahead (rolling out candidate
        decisions). Not the environment's simulator; the agent only ever
        sees the state through the observation.
    """

    def __init__(self, predictor: t.Optional[Predictor] = None,
                 optimizer: t.Optional[Optimizer] = None, *,
                 simulator: t.Optional["Simulator"] = None,
                 name: t.Optional[str] = None):
        if predictor is None and optimizer is None and type(self).act is Agent.act:
            raise ValueError("an Agent needs a predictor, an optimizer, or its own act()")
        parts = [p.name for p in (predictor, optimizer) if p is not None]
        super().__init__(name or (" → ".join(parts) if parts else type(self).__name__))
        self.predictor = predictor
        self.optimizer = optimizer
        self.simulator = simulator

    # -- the contract the run engine drives ---------------------------------------

    def bind(self, problem) -> "Agent":
        self.problem = problem
        for part in (self.predictor, self.optimizer):
            if part is not None and hasattr(part, "bind"):
                part.bind(problem)
        return self

    def fit(self, train) -> "Agent":
        """Fit the predictor on the training view. Optimizers are fit here too
        when they define ``fit``."""
        for part in (self.predictor, self.optimizer):
            if part is not None and hasattr(part, "fit"):
                part.fit(train)
        return self

    def forecast(self, obs) -> t.Optional[Forecast]:
        return self.predictor.predict(obs) if self.predictor is not None else None

    def act(self, obs) -> Decision:
        """The decision for this origin."""
        fc = self.forecast(obs)
        if self.optimizer is None:
            if fc is None:
                raise ValueError(f"{self.name}: no predictor and no optimizer; override act()")
            return fc
        return self.optimizer.decide(fc, obs)

    def predict(self, obs) -> pd.DataFrame:
        """Alias of :meth:`act`, so an Agent fits wherever a Predictor does."""
        return self.act(obs)

    def learn(self, record: "SettlementRecord") -> None:
        """Called with every settled decision. Override for online learning;
        the default forwards to the parts that define ``learn``."""
        for part in (self.predictor, self.optimizer):
            if part is not None and hasattr(part, "learn"):
                part.learn(record)

    def __repr__(self):
        return f"Agent({self.name!r})"
