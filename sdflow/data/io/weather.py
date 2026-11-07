"""rebase-weather API-backed fields: archived NWP runs as sdflow Fields.

The `Rebase weather API <https://api.rebase.energy>`_ serves point forecasts
from operational NWP models. Its ``/weather/v2/point/historical`` endpoint
returns the *archive of past runs* — every forecast as it was issued — which
is what a leak-free backtest needs: at each origin the feed serves the newest
run already disseminated, never a reanalysis of what actually happened.

This module turns those runs into :class:`~sdflow.data.field.Field` s of kind
``forecast``, indexed by ``(issue_time, valid_time)``, with the same
local parquet cache as :mod:`sdflow.data.io.grid` so problems built on live
data stay cheap, reproducible and runnable offline (hillclimb candidate
sandboxes read the cache and never need the API key).

Environment:

* ``SDFLOW_WEATHER_API_KEY`` (or ``REBASE_API_KEY``) — API key; only needed
  on a cache miss.
* ``SDFLOW_WEATHER_CACHE`` — cache root (default ``~/.cache/sdflow/weather``).
* ``SDFLOW_WEATHER_OFFLINE=1`` — forbid network; a cache miss becomes an error.

Variable names are the API's own (``WindSpeed:100``, ``Temperature``,
``WindDirection:10``, …). A ``:`` is not legal in a parquet column name on
every engine and reads badly in a feature matrix, so columns are emitted as
``<point>__<variable>`` with ``:`` folded to ``_`` — e.g.
``ostrobothnia__WindSpeed_100``.
"""

from __future__ import annotations

import os
import time
import typing as t
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..field import ISSUE_LEVEL, VALID_LEVEL, Field

BASE_URL = os.environ.get("SDFLOW_WEATHER_BASE_URL", "https://api.rebase.energy")

#: Wind-power predictors, in the order a Nordic wind model usually wants them.
#: 100 m is hub height for modern onshore turbines; 10 m is kept because it is
#: the variable every NWP model is best verified against.
#:
#: Verified against the live API (2026-08-21): only ``:10`` and ``:100`` are
#: real levels. ``WindSpeed:150`` returns the 10 m values verbatim rather than
#: erroring, so asking for it buys a duplicate column, not a higher level; a
#: bare ``WindSpeed``/``WindDirection``/``Temperature`` is ``:10``/``:10``/``:2``.
#: ``AirDensity``, ``WindGust`` and every pressure variable 500 — density has
#: to be approximated from temperature, since no pressure field is served.
#: Precipitation earns its place this far north: wet snow near 0 degC is what
#: ices Finnish rotors, and iced turbines are the largest single source of
#: winter forecast error.
WIND_VARIABLES = (
    "WindSpeed:100",
    "WindSpeed:10",
    "WindDirection:100",
    "WindDirection:10",
    "Temperature",
    "RelativeHumidity",
    "TotalPrecipitation",
)

#: Dissemination delay per model: how long after its nominal run hour the run
#: is actually queryable. Used as the field's ``availability_lag``, so an
#: origin only ever sees runs that had really landed by then.
MODEL_AVAILABILITY_LAG = {
    "DWD_ICON-EU": "4h",
    "NCEP_GFS": "5h",
    "MetNo_HIRESMEPS": "3h",
    "MetOffice_GlobalHiRes": "7h",
    "ECMWF_ERA5": "0h",  # reanalysis — not for operational backtests
}
DEFAULT_AVAILABILITY_LAG = "5h"

#: Bump when the request shape changes; part of every cache filename.
CACHE_VERSION = 2


@dataclass(frozen=True)
class Point:
    """One NWP grid point, named so its columns stay readable.

    ``weight`` lets a caller carry capacity weights alongside the coordinates
    (a zone-level aggregate is a capacity-weighted mean of its points); this
    module only passes it through — it never aggregates on its own.
    """

    name: str
    latitude: float
    longitude: float
    weight: float = 1.0


def cache_root() -> Path:
    return Path(os.environ.get("SDFLOW_WEATHER_CACHE",
                               Path.home() / ".cache" / "sdflow" / "weather"))


def _offline() -> bool:
    return os.environ.get("SDFLOW_WEATHER_OFFLINE", "") not in ("", "0")


def _api_key() -> t.Optional[str]:
    return os.environ.get("SDFLOW_WEATHER_API_KEY") or os.environ.get("REBASE_API_KEY")


def _utc(ts) -> pd.Timestamp:
    """Timestamps compare against a tz-aware valid_time index, so naive input
    (a bare ``"2026-01-01"``) is read as UTC rather than raising."""
    ts = pd.Timestamp(ts)
    return ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")


def _column(point: str, variable: str) -> str:
    return f"{point}__{variable.replace(':', '_')}"


def _get_json(url: str, params: dict, api_key: str, retries: int = 3):
    import requests

    for attempt in range(retries):
        resp = requests.get(url, params=params,
                            headers={"Authorization": api_key}, timeout=300)
        if resp.status_code in (429, 502, 503, 504) and attempt < retries - 1:
            time.sleep(2 ** attempt * (int(resp.headers.get("Retry-After", 0)) or 2))
            continue
        resp.raise_for_status()
        return resp.json()
    raise RuntimeError("unreachable")


def fetch_point_runs(point: Point, model: str, start, end, *,
                     variables: t.Sequence[str] = WIND_VARIABLES,
                     reference_time_freq: str = "24H",
                     forecast_horizon: str = "full",
                     api_key: t.Optional[str] = None) -> pd.DataFrame:
    """Fetch archived runs for one point over ``[start, end)``.

    Returns a frame indexed by ``(issue_time, valid_time)`` (both tz-aware
    UTC) with one column per variable, named ``<point>__<variable>``. Fetched
    in ~30-day chunks: the archive endpoint returns every run in the window,
    so a two-year request in one call is both slow and easy to time out.

    Do NOT pass ``provider=OM`` here. It makes the endpoint ignore
    ``reference-time-freq`` and ``forecast-horizon`` and return a stitched
    near-analysis instead — 3-hourly runs truncated to horizons 0-2h, which
    silently produces a dataset with no day-ahead forecasts in it at all.
    Without it, ``reference-time-freq=24H`` returns one 00Z run per day
    carrying horizons 0..120 (hourly to +78, 3-hourly after), which is what
    a day-ahead problem needs. Verified against the live API 2026-08-21.
    """
    key = api_key or _api_key()
    if key is None:
        raise RuntimeError(
            f"no API key for the rebase weather API (needed to fetch "
            f"{model} at {point.name}); set SDFLOW_WEATHER_API_KEY "
            f"(or REBASE_API_KEY)"
        )
    start, end = _utc(start), _utc(end)

    frames = []
    chunk_start = start
    while chunk_start < end:
        chunk_end = min(chunk_start + pd.Timedelta(days=30), end)
        payload = _get_json(
            f"{BASE_URL}/weather/v2/point/historical",
            {
                "model": model,
                "start-date": chunk_start.strftime("%Y-%m-%d"),
                "end-date": chunk_end.strftime("%Y-%m-%d"),
                "reference-time-freq": reference_time_freq,
                "forecast-horizon": forecast_horizon,
                "latitude": str(point.latitude),
                "longitude": str(point.longitude),
                "variables": ",".join(variables),
            },
            key,
        )
        frames.append(_runs_frame(payload, point, variables))
        chunk_start = chunk_end

    out = pd.concat(frames)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    valid = out.index.get_level_values(VALID_LEVEL)
    return out[(valid >= start) & (valid < end)]


def _runs_frame(payload, point: Point, variables: t.Sequence[str]) -> pd.DataFrame:
    """The API's parallel-array payload as an (issue_time, valid_time) frame."""
    record = payload[0] if isinstance(payload, list) else payload
    frame = pd.DataFrame(record)
    missing = [c for c in ("ref_datetime", "valid_datetime") if c not in frame.columns]
    if missing:
        raise RuntimeError(
            f"weather API response for {point.name} lacks {missing} — got "
            f"columns {list(frame.columns)}"
        )
    frame[ISSUE_LEVEL] = pd.to_datetime(frame.pop("ref_datetime"), utc=True)
    frame[VALID_LEVEL] = pd.to_datetime(frame.pop("valid_datetime"), utc=True)
    frame = frame.set_index([ISSUE_LEVEL, VALID_LEVEL])
    # The API may return None for a variable it could not serve; keep the
    # column (shape stays stable across chunks) but make it numeric NaN.
    for variable in variables:
        if variable not in frame.columns:
            frame[variable] = float("nan")
    frame = frame[list(variables)].apply(pd.to_numeric, errors="coerce")
    frame.columns = [_column(point.name, v) for v in variables]
    return frame


def load_point_runs(point: Point, model: str, start, end, *,
                    variables: t.Sequence[str] = WIND_VARIABLES,
                    reference_time_freq: str = "24H",
                    forecast_horizon: str = "full") -> pd.DataFrame:
    """Cache-first :func:`fetch_point_runs`: hit the API only on a cache miss.

    Keyed by every argument that changes the data, so a problem with a pinned
    ``data_end`` always reads identical runs — and sandboxed evaluations need
    neither network nor the API key.
    """
    start, end = _utc(start), _utc(end)
    # CACHE_VERSION guards against a change in the *call shape* (which params
    # we send) rather than in its arguments: dropping provider=OM changed the
    # data completely while every key component stayed identical, and a stale
    # parquet was served instead. Bump it whenever the request changes.
    stem = (f"{point.name}_{point.latitude:.3f}_{point.longitude:.3f}_{model}_"
            f"{start:%Y%m%dT%H%M}_{end:%Y%m%dT%H%M}_"
            f"{reference_time_freq}_{forecast_horizon}_{len(variables)}v"
            f"_v{CACHE_VERSION}")
    path = cache_root() / f"{stem}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    if _offline():
        raise RuntimeError(
            f"SDFLOW_WEATHER_OFFLINE is set and {path} is not cached — "
            f"pre-warm the cache by loading the problem once with network access"
        )
    frame = fetch_point_runs(point, model, start, end, variables=variables,
                             reference_time_freq=reference_time_freq,
                             forecast_horizon=forecast_horizon, api_key=None)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    frame.to_parquet(tmp)
    tmp.replace(path)
    return frame


def weather_field(name: str, points: t.Sequence[Point], model: str, start, end, *,
                  variables: t.Sequence[str] = WIND_VARIABLES,
                  reference_time_freq: str = "24H",
                  forecast_horizon: str = "full",
                  availability_lag: t.Optional[str] = None) -> Field:
    """Archived NWP runs across ``points`` as one ``forecast`` Field.

    Points are joined on ``(issue_time, valid_time)``, so the field carries
    ``len(points) * len(variables)`` columns. ``availability_lag`` defaults to
    the model's dissemination delay — the reason an origin at 09:00 UTC sees
    the 00Z run and not the 06Z one.
    """
    if not points:
        raise ValueError(f"weather_field {name!r}: no points given")
    lag = availability_lag or MODEL_AVAILABILITY_LAG.get(model, DEFAULT_AVAILABILITY_LAG)
    frames = [
        load_point_runs(point, model, start, end, variables=variables,
                        reference_time_freq=reference_time_freq,
                        forecast_horizon=forecast_horizon)
        for point in points
    ]
    frame = pd.concat(frames, axis=1).sort_index()
    return Field(
        name=name,
        frame=frame,
        kind="forecast",
        availability_lag=lag,
        description=(
            f"{model} point forecasts at {len(points)} location(s) "
            f"({', '.join(p.name for p in points)}), variables "
            f"{', '.join(variables)}; rebase weather API archive, runs every "
            f"{reference_time_freq}, disseminated {lag} after issue"
        ),
    )
