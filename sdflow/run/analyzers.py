"""Analyzers: observers of a run, fed each settlement as it lands.

The backtrader idea — one evaluation loop, many pluggable observers. Analyzers
never influence the run; they accumulate diagnostics and report in
``Result.analysis``.
"""

from __future__ import annotations

from abc import ABC
import typing as t

import numpy as np
import pandas as pd

from ..problems.metrics import quantile_columns


class Analyzer(ABC):
    """Base analyzer. ``setup`` receives the run context; ``on_settlement``
    each scored origin; ``finalize`` returns the diagnostics dict."""

    @property
    def name(self) -> str:
        return type(self).__name__

    def setup(self, feed, target_field, target_column, objective) -> None:
        self.feed = feed
        self.target_field = target_field
        self.target_column = target_column
        self.objective = objective

    def on_settlement(self, record) -> None:  # pragma: no cover - interface
        pass

    def finalize(self) -> dict:  # pragma: no cover - interface
        return {}


class PersistenceSkill(Analyzer):
    """Scores a persistence baseline (last knowable value at each origin,
    broadcast over its targets) and reports the model's skill against it.
    A model that can't beat this hasn't learned anything."""

    def __init__(self):
        self._baseline_scores: t.List[float] = []
        self._model_scores: t.List[float] = []
        self._weights: t.List[int] = []
        self._disabled = False

    def setup(self, feed, target_field, target_column, objective) -> None:
        super().setup(feed, target_field, target_column, objective)
        # Env-settled objectives (trading revenue) can't score a synthetic
        # baseline from (actuals, prediction) alone — opt out cleanly.
        self._disabled = (getattr(objective.metric, "settled_by_env", False)
                          or target_field is None)

    def on_settlement(self, record) -> None:
        if self._disabled or record.actuals is None:
            return
        hist = self.feed.history(record.origin.asof, self.target_field)
        col = hist[record.origin.column or self.target_column].dropna()
        if col.empty:
            return
        # Match the submission's output shape: for quantile predictions the
        # baseline is the point-mass persistence (last value at every level).
        levels = quantile_columns(record.prediction) or ["point"]
        baseline = pd.DataFrame({lvl: col.iloc[-1] for lvl in levels},
                                index=record.origin.target_index)
        n = int((record.actuals.notna()).sum())
        if n == 0:
            return
        self._baseline_scores.append(self.objective.calculate(record.actuals, baseline))
        self._model_scores.append(record.score)
        self._weights.append(n)

    def finalize(self) -> dict:
        if not self._weights:
            return {}
        w = np.asarray(self._weights, dtype=float)
        base = float(np.average(self._baseline_scores, weights=w))
        model = float(np.average(self._model_scores, weights=w))
        beats = self.objective.is_better(model, base)
        out = {"persistence_score": base, "model_score": model, "beats_persistence": bool(beats)}
        if base != 0:
            out["skill"] = float(1.0 - model / base) if self.objective.lower_is_better \
                else float(model / base - 1.0)
        return out


class QuantileCalibration(Analyzer):
    """Empirical coverage per predicted quantile. A calibrated q10 should have
    ~10% of actuals below it; large gaps mean the intervals are lying."""

    def __init__(self):
        self._below: t.Dict[float, int] = {}
        self._total: t.Dict[float, int] = {}

    def on_settlement(self, record) -> None:
        levels = quantile_columns(record.prediction)
        if not levels or record.actuals is None:
            return
        y = record.actuals
        for q in levels:
            pred = record.prediction[q].reindex(y.index)
            ok = y.notna() & pred.notna()
            self._below[q] = self._below.get(q, 0) + int((y[ok] <= pred[ok]).sum())
            self._total[q] = self._total.get(q, 0) + int(ok.sum())

    def finalize(self) -> dict:
        out = {}
        for q in sorted(self._total):
            if self._total[q]:
                out[f"coverage_q{int(round(q * 100)):02d}"] = self._below[q] / self._total[q]
        return out


class RegimeBreakdown(Analyzer):
    """Score and bias split by a regime read off one field, post hoc.

    Many problems have hours that matter more than others -- icing hours for a wind
    farm, peak hours for a grid, storm hours for a coastline -- and a single average
    hides whether a model is good *there*. This analyzer labels each scored target
    time by the value of ``field[column]`` at that time (``>= threshold`` is the first
    label, the rest the second; or ``bins`` edges for several) and reports, per label,
    the objective's per-timestamp error averaged over those hours, the number of hours,
    and the mean bias (actual minus the median or point prediction).

    The regime is read from the dataset after the fact, not through the feed's
    knowledge clock: it describes what happened, which is what a diagnostic should
    use, and it never reaches the model.
    """

    def __init__(self, field: str, column: str, *, threshold: t.Optional[float] = None,
                 labels: t.Sequence[str] = ("above", "below"),
                 bins: t.Optional[t.Sequence[float]] = None):
        if (threshold is None) == (bins is None):
            raise ValueError("RegimeBreakdown takes exactly one of threshold= or bins=")
        if bins is not None and len(labels) != len(bins) - 1:
            raise ValueError("labels must have one entry per bin")
        if threshold is not None and len(labels) != 2:
            raise ValueError("a threshold has two labels: at-or-above, below")
        self.field, self.column = field, column
        self.threshold, self.bins, self.labels = threshold, bins, tuple(labels)
        self._err: t.Dict[str, float] = {}
        self._bias: t.Dict[str, float] = {}
        self._n: t.Dict[str, int] = {}
        self._regime: t.Optional[pd.Series] = None

    @property
    def name(self) -> str:
        return "RegimeBreakdown"

    def setup(self, feed, target_field, target_column, objective) -> None:
        super().setup(feed, target_field, target_column, objective)
        frame = feed.dataset.field(self.field).frame
        values = frame[self.column]
        if self.threshold is not None:
            labels = np.where(values.to_numpy(float) >= self.threshold, self.labels[0], self.labels[1])
            regime = pd.Series(labels, index=values.index, dtype=object)
        else:
            regime = pd.cut(values, bins=list(self.bins), labels=list(self.labels), include_lowest=True)
            regime = regime.astype(object)
        regime[values.isna()] = None
        self._regime = regime[~regime.index.duplicated(keep="last")]

    def _median(self, prediction: pd.DataFrame) -> pd.Series:
        levels = quantile_columns(prediction)
        if levels:
            closest = min(levels, key=lambda q: abs(q - 0.5))
            return prediction[closest]
        return prediction.iloc[:, 0]

    def on_settlement(self, record) -> None:
        if record.actuals is None or self._regime is None:
            return
        y = record.actuals.dropna()
        if y.empty:
            return
        errors = pd.Series(np.asarray(self.objective.elementwise(y, record.prediction), dtype=float), index=y.index)
        bias = y - self._median(record.prediction).reindex(y.index)
        regime = self._regime.reindex(y.index)
        for label in self.labels:
            mask = (regime == label) & errors.notna()
            n = int(mask.sum())
            if n == 0:
                continue
            self._err[label] = self._err.get(label, 0.0) + float(errors[mask].sum())
            self._bias[label] = self._bias.get(label, 0.0) + float(bias[mask].fillna(0.0).sum())
            self._n[label] = self._n.get(label, 0) + n

    def finalize(self) -> dict:
        out: dict = {}
        total = sum(self._n.values())
        for label in self.labels:
            n = self._n.get(label, 0)
            if n == 0:
                continue
            out[f"{label}_score"] = self._err[label] / n
            out[f"{label}_bias"] = self._bias[label] / n
            out[f"{label}_hours"] = n
            out[f"{label}_share"] = n / total if total else 0.0
        return out


def default_analyzers(model=None) -> t.List[Analyzer]:
    analyzers: t.List[Analyzer] = [PersistenceSkill()]
    if model is None or getattr(model, "output_kind", "point") == "quantiles":
        analyzers.append(QuantileCalibration())
    return analyzers
