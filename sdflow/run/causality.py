"""Truncation invariance: the leak test that needs no theory of the model.

For every origin ``t``, the decision an agent makes at ``t`` must be the same
whether or not the dataset physically contains anything that was not knowable
at ``t``::

    decide(data[:t], t) == decide(data, t)

If deleting the future changes a decision, the agent (or a feature, or a
third-party library it calls) read the future. The feed makes that hard by
construction; this check catches what construction cannot: a model that keeps
a reference to the raw dataset, global state carried across runs, an indicator
computed on the whole series. It is property-based (sampled origins, exact
comparison), not a proof, and it is cheap enough to run on every benchmark.

The check is only as strong as the problem's ``make_env``: it must build its
environment from ``problem.load_dataset()`` and ``problem.origins(split)``, so
that a problem with a truncated dataset yields a truncated feed. The check
verifies this and refuses to report a pass otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field as _dc_field, replace
import typing as t

import pandas as pd

from ..data import Dataset, Field
from ..data.field import ISSUE_LEVEL
from ..problems.problem import Problem
from ..problems.schedule import IssueSchedule, Origin

__all__ = ["truncate_dataset", "truncation_invariance", "assert_truncation_invariant",
           "TruncationReport"]


def truncate_dataset(dataset: Dataset, asof) -> Dataset:
    """A copy of ``dataset`` with everything not knowable at ``asof`` removed.

    Actual fields keep rows with ``t + availability_lag <= asof`` (bitemporal
    ones: rows whose knowledge time is ``<= asof``); forecast fields keep runs
    with ``issue + availability_lag <= asof``; static fields are unchanged.
    """
    asof = pd.Timestamp(asof)
    fields = {}
    for name, f in dataset.fields.items():
        if f.kind == "actual":
            tz = f.frame.index.tz
            cut = asof.tz_localize(tz) if tz is not None and asof.tz is None else asof
            if f.knowledge_col is not None:
                keep = f.frame[f.knowledge_col] <= cut
                frame = f.frame[keep.to_numpy()]
            else:
                frame = f.frame.loc[:cut - f.availability_lag]
            fields[name] = Field(name, frame.copy(), kind="actual",
                                 availability_lag=f.availability_lag,
                                 knowledge_col=f.knowledge_col, description=f.description)
        elif f.kind == "forecast":
            issues = f.frame.index.get_level_values(ISSUE_LEVEL)
            cut = asof.tz_localize(issues.tz) if issues.tz is not None and asof.tz is None else asof
            frame = f.frame[(issues + f.availability_lag) <= cut]
            fields[name] = Field(name, frame.copy(), kind="forecast",
                                 availability_lag=f.availability_lag, description=f.description)
        else:
            fields[name] = f
    return Dataset(name=dataset.name, description=dataset.description,
                   collection=dataset.collection, fields=fields)


@dataclass
class TruncationReport:
    """Outcome of :func:`truncation_invariance`."""

    problem: str
    model: str
    checked: t.List[pd.Timestamp] = _dc_field(default_factory=list)
    violations: t.List[t.Tuple[pd.Timestamp, str]] = _dc_field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.violations

    def __str__(self):
        head = (f"truncation invariance of {self.model!r} on {self.problem!r}: "
                f"{len(self.checked)} origins checked, {len(self.violations)} violations")
        if not self.violations:
            return head
        lines = [head]
        for asof, msg in self.violations:
            first = msg.strip().splitlines()[0] if msg.strip() else ""
            lines.append(f"  {asof}: {first}")
        return "\n".join(lines)


def _decisions(problem: Problem, model, split: str, stop_at: pd.Timestamp,
               expect_dataset: t.Optional[Dataset] = None) -> t.Dict[pd.Timestamp, pd.DataFrame]:
    """Drive the loop and return the decision at each origin up to ``stop_at``
    (inclusive), without settling the origin at ``stop_at``."""
    env = problem.env(split)
    if expect_dataset is not None and env.feed.dataset is not expect_dataset:
        raise RuntimeError(
            f"problem {problem.name!r}: make_env did not build its feed from "
            f"problem.load_dataset(), so a truncated dataset has no effect and the "
            f"truncation check cannot run"
        )
    obs, info = env.reset()
    if hasattr(model, "bind"):
        model.bind(problem)
    if "train" in info and hasattr(model, "fit"):
        model.fit(info["train"])
    act = model.act if hasattr(model, "act") else model.predict
    learn = getattr(model, "learn", None)

    out: t.Dict[pd.Timestamp, pd.DataFrame] = {}
    while obs is not None:
        action = act(obs)
        out[obs.asof] = action.copy() if isinstance(action, pd.DataFrame) else action
        if obs.asof >= stop_at:
            break
        obs, _, terminated, _, info = env.step(action)
        if learn is not None:
            for record in info["settled"]:
                learn(record)
        if terminated:
            break
    return out


def truncation_invariance(problem: Problem, make_model: t.Callable[[], t.Any], *,
                          split: str = "validation", origins: t.Optional[t.Sequence[Origin]] = None,
                          k: int = 6) -> TruncationReport:
    """Check that the decision at each sampled origin is bit-identical when the
    dataset is physically truncated at that origin.

    ``make_model`` returns a fresh model each call (a class, or a lambda); the
    model must be deterministic given its inputs (seed it) for the comparison
    to mean anything. ``origins`` defaults to ``k`` origins spread over the split.
    """
    full_origins = problem.origins(split)
    if not full_origins:
        raise ValueError(f"no origins in split {split!r}")
    if origins is None:
        step = max(1, len(full_origins) // k)
        origins = full_origins[::step][:k]
    stop_at = max(o.asof for o in origins)

    full_model = make_model()
    full = _decisions(problem, full_model, split, stop_at)
    report = TruncationReport(problem=problem.name,
                              model=getattr(full_model, "name", type(full_model).__name__))

    dataset = problem.load_dataset()
    for origin in origins:
        truncated = truncate_dataset(dataset, origin.asof)
        upto = [o for o in full_origins if o.asof <= origin.asof]
        trimmed = replace(problem, dataset=truncated, schedule=IssueSchedule.explicit(upto),
                          _dataset_cache=None)
        partial = _decisions(trimmed, make_model(), split, origin.asof, expect_dataset=truncated)
        report.checked.append(origin.asof)
        try:
            pd.testing.assert_frame_equal(full[origin.asof], partial[origin.asof],
                                          check_exact=True)
        except AssertionError as exc:
            report.violations.append((origin.asof, str(exc)))
    return report


def assert_truncation_invariant(problem: Problem, make_model: t.Callable[[], t.Any],
                                **kwargs) -> TruncationReport:
    """:func:`truncation_invariance`, raising ``AssertionError`` on any violation."""
    report = truncation_invariance(problem, make_model, **kwargs)
    if not report.ok:
        details = "\n\n".join(f"origin {asof}:\n{msg}" for asof, msg in report.violations)
        raise AssertionError(f"{report}\n\n{details}")
    return report
