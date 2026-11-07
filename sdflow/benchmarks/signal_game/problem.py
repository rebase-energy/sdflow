"""The signal game: a hidden state, noisy sensors, a place to dig.

A treasure lies at one of ``W`` places. Each episode, ``K`` sensors report:
*signal* sensors each give one bit of the place's address, flipped with
probability ``flip``; *noise* sensors emit one of ``n_symbols`` symbols
uniformly, so they have the **highest** outcome entropy and carry **no**
information. The agent reads the sensors and digs; the payoff is 1 at the right
place and 0 elsewhere. The true place is revealed after the dig, so an agent
can learn across episodes which sensors mean anything.

This is the one-shot form of the game in `intelligence-theory
<https://github.com/sebaheg/intelligence-theory>`_ (every sensor is read; the
query budget of the original is a later addition that needs sub-steps within an
origin). It is here for two reasons. It is the first problem in sdflow with no
energy in it, which keeps the core honest. And it has exact answers: the
expected payoff of acting on the prior alone (chance), of the Bayes-optimal
reader who knows the channel model (the ceiling an informed agent can reach),
and of knowing the place outright (1.0). Those are the reference scores, so a
score on this problem is a measurement against known truth, which makes the
problem a correctness check of the loop, the feed and the settlement as much
as a benchmark of agents.

Timing: an episode is one origin. Its sensor readings are stamped at the
origin's clock time (knowable when the agent acts); the hidden place is stamped
at the target time, one hour later, so it is revealed exactly when the next
origin opens and the episode settles then, through the same delayed-settlement
rule every score in sdflow follows.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import itertools
import typing as t

import numpy as np
import pandas as pd

import sdflow as sd
from sdflow.data import DataFeed, Dataset, Field
from sdflow.envs import ColumnsAction, DecisionEnv, ScoreSettlement
from sdflow.problems.metrics import Metric, _point, _to_series

SENSORS_FIELD = "sensors"
HIDDEN_FIELD = "hidden"
DIG_COL = "dig"
START = pd.Timestamp("2000-01-01", tz="UTC")
N_TRAIN, N_VALIDATION, N_HOLDOUT = 500, 500, 500


# -- the task tables -----------------------------------------------------------------

def _bits(x: int, n: int) -> t.List[int]:
    return [(x >> (n - 1 - i)) & 1 for i in range(n)]


@dataclass(frozen=True)
class SignalTask:
    """``likelihood[k, h, o] = P(outcome o | sensor k, place h)``; ``payoff[h, a]``."""

    likelihood: np.ndarray      # (K, W, M)
    payoff: np.ndarray          # (W, A)
    labels: t.Tuple[str, ...]   # which sensors carry signal (hidden from agents)
    flip: float

    @property
    def n_places(self) -> int:
        return self.likelihood.shape[1]

    @property
    def n_sensors(self) -> int:
        return self.likelihood.shape[0]

    @property
    def n_symbols(self) -> int:
        return self.likelihood.shape[2]

    @property
    def prior(self) -> np.ndarray:
        return np.full(self.n_places, 1.0 / self.n_places)

    def chance_payoff(self) -> float:
        """Acting on the prior alone."""
        return float(np.max(self.prior @ self.payoff))

    def max_payoff(self) -> float:
        """Knowing the place."""
        return float(self.prior @ np.max(self.payoff, axis=1))

    def bayes_payoff(self) -> float:
        """Exact expected payoff of the Bayes-optimal reader of all sensors:
        Σ_o P(o) · max_a Σ_h P(h | o) payoff[h, a], enumerated over outcomes."""
        K, W, M = self.likelihood.shape
        prior = self.prior
        total = 0.0
        # Enumerate the first few sensors' outcomes in an outer loop and vectorise
        # the rest, so W=16 with ten sensors stays small in memory.
        outer = min(K, max(0, K - 8))
        for head in itertools.product(range(M), repeat=outer):
            lik = prior.copy()
            for k, o in enumerate(head):
                lik = lik * self.likelihood[k, :, o]
            lik = lik[None, :]                                     # (1, W)
            for k in range(outer, K):
                lik = (lik[:, None, :] * self.likelihood[k].T[None, :, :]).reshape(-1, W)
            p_o = lik.sum(axis=1)                                  # (n, )
            value = (lik @ self.payoff).max(axis=1)                # Σ_h P(h, o) payoff[h, a], best a
            total += float(value.sum())
            del p_o
        return total


def build_task(n_places: int = 8, n_signal: int = 3, n_noise: int = 3, flip: float = 0.1,
               n_symbols: int = 4, seed: int = 0) -> SignalTask:
    """Port of ``intelligence_theory.tasks.signal_game``: sensor order is shuffled so
    an agent cannot read a sensor's type off its index."""
    if n_symbols < 2:
        raise ValueError("need at least two symbols")
    rng = np.random.default_rng(seed)
    nbits = max(1, int(np.ceil(np.log2(n_places))))
    K = n_signal + n_noise
    L = np.zeros((K, n_places, n_symbols))
    labels: t.List[str] = []
    for j in range(n_signal):
        bit = j % nbits
        for h in range(n_places):
            b = _bits(h, nbits)[bit]
            L[j, h, b] = 1.0 - flip
            L[j, h, 1 - b] = flip
        labels.append(f"bit{bit}")
    for j in range(n_signal, K):
        L[j, :, :] = 1.0 / n_symbols
        labels.append("noise")
    perm = rng.permutation(K)
    return SignalTask(L[perm], np.eye(n_places), tuple(labels[i] for i in perm), flip)


# -- the dataset ------------------------------------------------------------------------

def sensor_columns(task: SignalTask) -> t.List[str]:
    return [f"sensor_{k}" for k in range(task.n_sensors)]


def build_dataset(task: SignalTask, n_episodes: int, seed: int = 0) -> Dataset:
    """``n_episodes`` hourly episodes from :data:`START`: sensors stamped at the
    origin (``T - 1h``), the hidden place stamped at the target ``T``."""
    rng = np.random.default_rng(seed)
    targets = pd.date_range(START + pd.Timedelta("1h"), periods=n_episodes, freq="1h")
    places = rng.choice(task.n_places, size=n_episodes, p=task.prior)
    readings = np.empty((n_episodes, task.n_sensors), dtype=int)
    for k in range(task.n_sensors):
        cdf = task.likelihood[k].cumsum(axis=1)               # (W, M)
        u = rng.random(n_episodes)
        readings[:, k] = (u[:, None] > cdf[places]).sum(axis=1)
    sensors = pd.DataFrame(readings, index=targets - pd.Timedelta("1h"), columns=sensor_columns(task))
    hidden = pd.DataFrame({"place": places}, index=targets)
    return Dataset(
        name="signal-game",
        description=__doc__,
        fields={
            SENSORS_FIELD: Field(SENSORS_FIELD, sensors, availability_lag="0h",
                                 description="sensor readings, knowable at the origin"),
            HIDDEN_FIELD: Field(HIDDEN_FIELD, hidden, availability_lag="0h",
                                description="the treasure's place, revealed at the target time"),
        },
    )


# -- the objective ------------------------------------------------------------------------

class TablePayoff(Metric):
    """``payoff[h, a]`` per episode, averaged. Higher is better."""

    aggregate = "mean"

    def __init__(self, payoff: np.ndarray):
        self.payoff = np.asarray(payoff, dtype=float)

    @property
    def name(self) -> str:
        return "Payoff"

    def elementwise(self, y_true, y_pred):
        y = _to_series(y_true)
        a = _point(y_pred).reindex(y.index)
        ok = y.notna() & a.notna()
        out = np.full(len(y), np.nan)
        out[ok.to_numpy()] = self.payoff[y[ok].to_numpy().astype(int), a[ok].to_numpy().astype(int)]
        return out


# -- the problem -----------------------------------------------------------------------

VARIANTS: t.Dict[str, dict] = {
    "w8": dict(n_places=8, n_signal=3, n_noise=3, flip=0.1),
    "w16": dict(n_places=16, n_signal=6, n_noise=4, flip=0.2),
}


@lru_cache(maxsize=None)
def task_for(variant: str) -> SignalTask:
    return build_task(**VARIANTS[variant])


def _episode_time(i: int) -> pd.Timestamp:
    return START + pd.Timedelta(hours=i + 1)


SPLITS = sd.Splits(
    train_end=_episode_time(N_TRAIN),
    validation=(_episode_time(N_TRAIN), _episode_time(N_TRAIN + N_VALIDATION)),
    holdout=(_episode_time(N_TRAIN + N_VALIDATION),
             _episode_time(N_TRAIN + N_VALIDATION + N_HOLDOUT)),
)


def make_env(problem: sd.Problem, split: str) -> DecisionEnv:
    return DecisionEnv(
        feed=DataFeed(problem.load_dataset()),
        origins=problem.origins(split),
        settlement=ScoreSettlement(problem.objective, HIDDEN_FIELD, "place"),
        objective=problem.objective,
        action_spec=ColumnsAction(DIG_COL, what="digs"),
        train_end=problem.splits.train_end,
        target_field=HIDDEN_FIELD,
        target_column="place",
    )


def reference_scores(task: SignalTask) -> t.List[sd.RefScore]:
    return [
        sd.RefScore(rank=1, team="oracle (knows the place)", score=task.max_payoff()),
        sd.RefScore(rank=2, team="bayes (knows the channel model)", score=round(task.bayes_payoff(), 6)),
        sd.RefScore(rank=3, team="chance (acts on the prior)", score=task.chance_payoff()),
    ]


def _problem(variant: str) -> sd.Problem:
    task = task_for(variant)
    return sd.Problem(
        name=f"signal-game:{variant}",
        dataset=lambda: build_dataset(task, N_TRAIN + N_VALIDATION + N_HOLDOUT, seed=0),
        make_env=make_env,
        objective=sd.Objective(TablePayoff(task.payoff), lower_is_better=False),
        schedule=sd.IssueSchedule.hourly(horizon="1h"),
        splits=SPLITS,
        description=__doc__,
        reference_scores=reference_scores(task),
    )


def list_problem_variants():
    return list(VARIANTS)


def get_problem_w8() -> sd.Problem:
    return _problem("w8")


def get_problem_w16() -> sd.Problem:
    return _problem("w16")
