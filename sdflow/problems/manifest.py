"""Problems declared in a dataset repo's ``rebase.yaml``, no Python package required.

A Rebase-compatible dataset repo already lists its fields in ``rebase.yaml`` (see
:meth:`~sdflow.data.dataset.Dataset.from_manifest`). A ``problems:`` section turns the repo
into a benchmark: each entry names the fields it uses, the target, the objective, the
issue schedule and the splits, and sdflow builds the :class:`~sdflow.problems.problem.Problem`
from it. Anyone with sdflow installed can then run ::

    >>> sdflow.load_problem("rb://dataset/rebase-energy/icing-no:odal")
    >>> sdflow.load_problem("/path/to/staged/repo:odal")        # a local directory

The reference is ``<root>:<variant>``, where ``<root>`` is an ``rb://dataset/<owner>/<name>``
URI, a bare ``<owner>/<name>`` Hub repo, or a directory. The variant may be left out when the
manifest declares exactly one problem.

Manifest schema (only the ``problems`` part; ``fields`` is the dataset manifest) ::

    problems:
      odal:
        description: Odal day-ahead output with icing, MAE in MW.
        fields: {power: odal/power, meps: odal/meps}    # problem field -> manifest key
        target: {field: power, column: power}
        objective: {metric: mae}                        # mae | mse | rmse | mape | pinball
        schedule: {daily: {at: "10:00", covers: ["12h", "35h"], target_freq: "1h"}}
        splits:
          train_end: "2024-10-01T00:00:00Z"
          validation: ["2024-10-01T00:00:00Z", "2025-04-01T00:00:00Z"]
          holdout: ["2025-10-01T00:00:00Z", "2026-01-01T00:00:00Z"]
        reference_scores: [{team: climatology, score: 23.4}]   # optional, ranked in order

``fields`` may also be a list of manifest keys; the problem then names each field by the
part after the last ``/`` (``odal/power`` becomes ``power``). A pinball objective takes
``quantiles``; the environment asks the predictor for exactly those. Only the fields a
problem lists are read, so a repo holding several farms costs one farm per load.
"""

from __future__ import annotations

import typing as t

from ..data.dataset import Dataset, read_manifest
from .metrics import (
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    MeanSquaredError,
    Metric,
    PinballLoss,
    RootMeanSquaredError,
)
from .objective import Objective
from .problem import Problem, RefScore, Splits
from .schedule import IssueSchedule

METRICS: t.Dict[str, t.Callable[..., Metric]] = {
    "mae": MeanAbsoluteError,
    "mse": MeanSquaredError,
    "rmse": RootMeanSquaredError,
    "mape": MeanAbsolutePercentageError,
    "pinball": PinballLoss,
}


class ManifestError(ValueError):
    """The manifest cannot be turned into a problem; the message says which key."""


def is_manifest_reference(name: str) -> bool:
    """Does ``name`` point at a repo or directory rather than a registered problem?

    Registered problems are ``package`` or ``package:variant`` and never contain a ``/``;
    a manifest reference always does (an ``rb://`` URI, ``owner/name`` or a path).
    """
    return "/" in name or "://" in name


def split_reference(name: str) -> t.Tuple[str, t.Optional[str]]:
    """``rb://dataset/o/n:odal`` -> ``("rb://dataset/o/n", "odal")``; no variant -> ``None``.

    The variant is the text after the last ``:`` only when that colon comes after the
    last ``/``, so the ``:`` of ``rb://`` and Windows drive letters are left alone.
    """
    root, sep, variant = name.rpartition(":")
    if sep and "/" not in variant and variant:
        return root, variant
    return name, None


def manifest_problems(root: str, token: t.Optional[str] = None) -> t.List[str]:
    """The problem names a repo or directory declares, as ``<manifest name>:<variant>``."""
    manifest, _ = read_manifest(root, token=token)
    return [f"{manifest.get('name', _basename(root))}:{v}" for v in (manifest.get("problems") or {})]


def load_manifest_problem(reference: str, token: t.Optional[str] = None) -> Problem:
    """Build the :class:`Problem` that ``<root>:<variant>`` declares."""
    root, variant = split_reference(reference)
    manifest, read_frame = read_manifest(root, token=token)
    problems = manifest.get("problems") or {}
    if not problems:
        raise ManifestError(f"{root} declares no problems (no `problems:` section in rebase.yaml)")
    if variant is None:
        if len(problems) != 1:
            raise ManifestError(
                f"{root} declares {len(problems)} problems; name one: "
                + ", ".join(f"{root}:{v}" for v in problems)
            )
        variant = next(iter(problems))
    if variant not in problems:
        raise ManifestError(f"{root} has no problem {variant!r}; known: {sorted(problems)}")
    spec = problems[variant] or {}
    name = f"{manifest.get('name', _basename(root))}:{variant}"
    return _build(name, spec, manifest, read_frame, root)


# -- pieces ------------------------------------------------------------------------------------


def _basename(root: str) -> str:
    return root.rstrip("/").rsplit("/", 1)[-1]


def _field_map(spec: dict, manifest: dict, name: str) -> t.Dict[str, str]:
    """Problem field name -> manifest key, validated against the manifest's ``fields``."""
    declared = spec.get("fields")
    if not declared:
        raise ManifestError(f"{name}: `fields` must list the manifest fields the problem uses")
    if isinstance(declared, list):
        mapping = {key.rsplit("/", 1)[-1]: key for key in declared}
    elif isinstance(declared, dict):
        mapping = dict(declared)
    else:
        raise ManifestError(f"{name}: `fields` must be a list or a mapping")
    known = manifest.get("fields") or {}
    missing = [key for key in mapping.values() if key not in known]
    if missing:
        raise ManifestError(f"{name}: fields {missing} are not in the manifest's `fields`")
    return mapping


def _objective(spec: dict, name: str) -> t.Tuple[Objective, t.Optional[t.Tuple[float, ...]]]:
    raw = spec.get("objective") or {}
    if isinstance(raw, str):
        raw = {"metric": raw}
    metric_name = str(raw.get("metric", "")).lower()
    if metric_name not in METRICS:
        raise ManifestError(f"{name}: objective.metric must be one of {sorted(METRICS)}, got {metric_name!r}")
    quantiles = raw.get("quantiles")
    if metric_name == "pinball":
        if not quantiles:
            raise ManifestError(f"{name}: a pinball objective needs `quantiles`")
        quantiles = tuple(float(q) for q in quantiles)
        metric = PinballLoss(quantiles)
    else:
        if quantiles:
            raise ManifestError(f"{name}: `quantiles` only applies to a pinball objective")
        metric = METRICS[metric_name]()
    return Objective(metric, lower_is_better=bool(raw.get("lower_is_better", True))), quantiles


def _schedule(spec: dict, name: str) -> IssueSchedule:
    raw = spec.get("schedule") or {}
    if "daily" in raw:
        d = raw["daily"] or {}
        covers = d.get("covers", ("1D", "2D"))
        if len(covers) != 2:
            raise ManifestError(f"{name}: schedule.daily.covers must be [start, end]")
        return IssueSchedule.daily(at=str(d.get("at", "09:00")), covers=(str(covers[0]), str(covers[1])),
                                   target_freq=str(d.get("target_freq", "1h")))
    raise ManifestError(f"{name}: schedule must be {{daily: {{at, covers, target_freq}}}}; got {sorted(raw)}")


def _splits(spec: dict, name: str) -> Splits:
    raw = spec.get("splits") or {}
    try:
        return Splits(train_end=raw["train_end"], validation=tuple(raw["validation"]), holdout=tuple(raw["holdout"]))
    except KeyError as exc:
        raise ManifestError(f"{name}: splits needs train_end, validation and holdout; missing {exc}") from None


def _reference_scores(spec: dict) -> t.List[RefScore]:
    raw = spec.get("reference_scores") or []
    if isinstance(raw, dict):
        raw = [{"team": team, "score": score} for team, score in raw.items()]
    scores = []
    for i, row in enumerate(raw, start=1):
        scores.append(RefScore(rank=int(row.get("rank", i)), team=str(row["team"]), score=float(row["score"])))
    return scores


def _build(name: str, spec: dict, manifest: dict, read_frame, root: str) -> Problem:
    mapping = _field_map(spec, manifest, name)
    target = spec.get("target") or {}
    target_field = target.get("field")
    if target_field not in mapping:
        raise ManifestError(f"{name}: target.field must be one of the problem's fields {sorted(mapping)}")
    target_column = target.get("column")
    objective, quantiles = _objective(spec, name)
    schedule = _schedule(spec, name)
    splits = _splits(spec, name)

    def load_dataset() -> Dataset:
        return Dataset.from_manifest_fields(manifest, read_frame, mapping, name=name)

    def make_env(problem: Problem, split: str):
        from ..data import DataFeed
        from ..envs import ForecastEnv

        return ForecastEnv(feed=DataFeed(problem.load_dataset()), origins=problem.origins(split),
                           target_field=target_field, objective=problem.objective,
                           target_column=target_column, train_end=problem.splits.train_end,
                           quantiles=quantiles)

    return Problem(
        name=name,
        dataset=load_dataset,
        make_env=make_env,
        objective=objective,
        schedule=schedule,
        splits=splits,
        description=spec.get("description") or manifest.get("description"),
        reference_scores=_reference_scores(spec),
    )


__all__ = ["ManifestError", "is_manifest_reference", "split_reference", "manifest_problems",
           "load_manifest_problem", "METRICS"]
