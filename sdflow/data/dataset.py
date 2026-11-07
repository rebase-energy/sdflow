"""Dataset: a named collection of :class:`Field` s (+ optional asset collection).

A dataset can be built in code (fields passed directly) or loaded lazily from a
Rebase-compatible Hugging Face repo via :meth:`Dataset.from_manifest`, where the
repo's ``rebase.yaml`` declares each field's parquet file and availability rule::

    name: heftcom2024
    description: Hybrid Energy Forecasting and Trading Competition 2024
    fields:
      wind_power:
        path: data/wind_power.parquet
        kind: actual
        availability_lag: 1h
      dwd_nwp:
        path: data/dwd_nwp.parquet
        kind: forecast            # parquet already has (issue_time, valid_time) index
"""

from __future__ import annotations

from dataclasses import dataclass, field as _dc_field
import typing as t
from pathlib import Path

import pandas as pd

from .field import Field

@dataclass
class Dataset:
    """Named fields + an optional collection of the entities the data describes
    (for energy problems an :mod:`energydatamodel` collection from :mod:`sdflow.energy`).

    ``fields`` maps name -> :class:`Field`. The legacy ``data`` dict-of-frames
    view is kept as a convenience (``dataset.data["wind_power"]`` returns the
    raw frame), but evaluation code must go through a
    :class:`~sdflow.data.feed.DataFeed`, never through ``data``.
    """

    name: str
    description: t.Optional[str] = None
    collection: t.Optional[t.Any] = None
    fields: t.Dict[str, Field] = _dc_field(default_factory=dict)

    def __post_init__(self):
        for key, f in list(self.fields.items()):
            if not isinstance(f, Field):
                # Accept raw frames for convenience; default availability.
                self.fields[key] = Field(name=key, frame=f)
            elif f.name != key:
                raise ValueError(f"fields[{key!r}] holds Field named {f.name!r}")

    # -- access ---------------------------------------------------------------

    def field(self, name: str) -> Field:
        try:
            return self.fields[name]
        except KeyError:
            raise KeyError(
                f"dataset {self.name!r} has no field {name!r}; "
                f"available: {sorted(self.fields)}"
            ) from None

    def add(self, field: Field) -> "Dataset":
        self.fields[field.name] = field
        return self

    @property
    def data(self) -> t.Dict[str, pd.DataFrame]:
        """Raw frames by field name (convenience view — not leak-safe)."""
        return {name: f.frame for name, f in self.fields.items()}

    @property
    def list_data(self):
        return list(self.fields.keys())

    # -- manifest loading -------------------------------------------------------

    @classmethod
    def from_manifest(cls, repo: str, token: t.Optional[str] = None) -> "Dataset":
        """Load every field of a Rebase-compatible repo's ``rebase.yaml``.

        ``repo`` is an ``rb://dataset/<owner>/<name>`` URI, a bare ``<owner>/<name>``
        (treated as a dataset repo) or a local directory holding a staged repo. Field
        frames are read with pandas via fsspec, so ``rb://`` caching and ``HF_TOKEN``
        handling apply.
        """
        manifest, read_frame = read_manifest(repo, token=token)
        mapping = {name: name for name in (manifest.get("fields") or {})}
        return cls.from_manifest_fields(manifest, read_frame, mapping,
                                        name=manifest.get("name", repo.rstrip("/").rsplit("/", 1)[-1]))

    @classmethod
    def from_manifest_fields(cls, manifest: dict, read_frame: t.Callable[[str], pd.DataFrame],
                             mapping: t.Mapping[str, str], name: str) -> "Dataset":
        """Build a dataset from a subset of a manifest's fields.

        ``mapping`` is field name in the dataset -> key in ``manifest["fields"]``;
        ``read_frame`` turns a manifest path into a frame (from :func:`read_manifest`).
        Only the listed fields are read.
        """
        specs = manifest.get("fields") or {}
        fields = {}
        for field_name, key in mapping.items():
            spec = specs[key]
            frame = read_frame(spec["path"])
            kind = spec.get("kind", "actual")
            if kind == "forecast" and not isinstance(frame.index, pd.MultiIndex):
                frame = frame.set_index([spec["issue_col"], spec["valid_col"]])
            fields[field_name] = Field(
                name=field_name, frame=frame, kind=kind,
                availability_lag=spec.get("availability_lag", "0h"),
                knowledge_col=spec.get("knowledge_col"),
                description=spec.get("description"),
            )
        return cls(name=name, description=manifest.get("description"), fields=fields)

    def __repr__(self):
        return f"Dataset({self.name!r}, fields={sorted(self.fields)})"


def read_manifest(root: str, token: t.Optional[str] = None) -> t.Tuple[dict, t.Callable[[str], pd.DataFrame]]:
    """Read ``rebase.yaml`` at ``root`` and return it with a reader for its files.

    ``root`` is an ``rb://dataset/<owner>/<name>`` URI, a bare ``<owner>/<name>`` Hub
    repo, or a local directory. The reader takes a path relative to the root (as the
    manifest's ``path`` entries are written) and returns the parquet as a frame.
    """
    import yaml

    local = Path(root).expanduser()
    if "://" not in root and local.is_dir():
        manifest = yaml.safe_load((local / "rebase.yaml").read_text())
        return manifest, lambda rel: pd.read_parquet(local / rel)

    import fsspec

    repo = root if root.startswith("rb://") else f"rb://dataset/{root}"
    repo = repo.rstrip("/")
    storage = {"token": token} if token else {}
    with fsspec.open(f"{repo}/rebase.yaml", "r", **storage) as fh:
        manifest = yaml.safe_load(fh)
    return manifest, lambda rel: pd.read_parquet(f"{repo}/{rel}", storage_options=storage or None)
