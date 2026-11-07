"""Fingrid open-data API-backed fields: Finnish TSO series as sdflow Fields.

`Fingrid <https://data.fingrid.fi>`_ publishes the Finnish system's own
measurements, several years deep — far deeper than the aggregated feeds. For
wind it serves three series that matter:

* ``wind`` (dataset 75) — wind power generation, MW. The forecast target.
* ``wind_capacity`` (dataset 268) — installed capacity used in Fingrid's own
  wind forecast, MW. Finland's fleet grew from ~2.3 GW in 2021 to ~9.4 GW in
  2026, so without this the target is not remotely stationary.
* ``wind_forecast_da`` (dataset 246) — Fingrid's own once-daily wind
  forecast, MW. The public number a Finnish trader already has, and therefore
  the bar any model has to clear. See the caveat in :func:`fetch_series`.

Same cache-first parquet contract as :mod:`sdflow.data.io.grid` and
:mod:`sdflow.data.io.weather`: warm it once with network access and every
later evaluation runs offline, with no key.

Environment:

* ``SDFLOW_FINGRID_API_KEY`` (or ``FINGRID_API_KEY`` / ``FINGRID_PRIMARY``)
* ``SDFLOW_FINGRID_CACHE`` — cache root (default ``~/.cache/sdflow/fingrid``).
* ``SDFLOW_FINGRID_OFFLINE=1`` — forbid network; a cache miss becomes an error.

Resolution changes mid-history: hourly through 2024, 15-minute from the
Nordic 15-minute MTU transition onward. Everything is resampled to a single
hourly grid (interval mean), so a series spanning the change stays uniform.
"""

from __future__ import annotations

import json
import os
import time
import typing as t
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

from ..field import Field

BASE_URL = os.environ.get("SDFLOW_FINGRID_BASE_URL", "https://data.fingrid.fi/api")

#: variable -> Fingrid dataset id.
DATASETS: t.Dict[str, int] = {
    "wind": 75,
    "wind_capacity": 268,
    "wind_forecast_da": 246,
}

#: Fingrid caps pageSize here; larger windows page via ``nextPage``.
MAX_PAGE_SIZE = 20_000

#: Fetch window per request. A year of 15-minute data is ~35k rows (2 pages);
#: 90 days keeps each call small enough to retry cheaply after a 429.
CHUNK_DAYS = 90

#: Publication delay for measured generation. Fingrid serves 75 within the
#: hour, but a settled value can be revised, so an hour of slack keeps the
#: backtest honest without materially changing a day-ahead problem.
DEFAULT_AVAILABILITY_LAG = "1h"

#: Bump when the request shape changes; part of every cache filename.
CACHE_VERSION = 1


def cache_root() -> Path:
    return Path(os.environ.get("SDFLOW_FINGRID_CACHE",
                               Path.home() / ".cache" / "sdflow" / "fingrid"))


def _offline() -> bool:
    return os.environ.get("SDFLOW_FINGRID_OFFLINE", "") not in ("", "0")


def _api_key() -> t.Optional[str]:
    return (os.environ.get("SDFLOW_FINGRID_API_KEY")
            or os.environ.get("FINGRID_API_KEY")
            or os.environ.get("FINGRID_PRIMARY"))


def _utc(ts) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")


def _get_json(params: dict, dataset: int, api_key: str, retries: int = 6) -> dict:
    """One API call. Fingrid rate-limits hard and says so in the body, so 429
    is retried with a widening backoff rather than treated as an error."""
    url = f"{BASE_URL}/datasets/{dataset}/data?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(
        url, headers={"x-api-key": api_key, "User-Agent": "sdflow"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            retryable = exc.code == 429 or exc.code in (502, 503, 504)
            if retryable and attempt < retries - 1:
                time.sleep(2 + 3 * attempt)
                continue
            raise
    raise RuntimeError("unreachable")


def _fetch_window(dataset: int, start: pd.Timestamp, end: pd.Timestamp,
                  api_key: str) -> list:
    """Every row in ``[start, end)``, following ``nextPage`` to the last page."""
    rows, page = [], 1
    while True:
        payload = _get_json(
            {
                "startTime": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "endTime": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "format": "json", "pageSize": MAX_PAGE_SIZE,
                "page": page, "sortOrder": "asc",
            },
            dataset, api_key,
        )
        rows.extend(payload.get("data", []))
        next_page = (payload.get("pagination") or {}).get("nextPage")
        if not next_page:
            return rows
        page = next_page


def fetch_series(variable: str, start, end, *,
                 api_key: t.Optional[str] = None) -> pd.DataFrame:
    """Fetch one variable over ``[start, end)`` as an hourly UTC frame.

    Rows are interval records (``startTime``/``endTime``/``value``) which may
    be hourly or 15-minute depending on the date; both are resampled to a
    single hourly grid by interval mean.

    Note on ``wind_forecast_da``: the API serves it by *valid* time with no
    issue time, so when a given value was published cannot be recovered. It
    is therefore safe as a reference series to score against, and NOT safe as
    a model input — treating it as knowable at a day-ahead origin assumes a
    publication time the API never states.
    """
    if variable not in DATASETS:
        raise KeyError(f"unknown variable {variable!r}; known: {sorted(DATASETS)}")
    key = api_key or _api_key()
    if key is None:
        raise RuntimeError(
            f"no API key for the Fingrid API (needed to fetch {variable}); set "
            f"SDFLOW_FINGRID_API_KEY (or FINGRID_API_KEY / FINGRID_PRIMARY)"
        )
    dataset = DATASETS[variable]
    start, end = _utc(start), _utc(end)

    rows: list = []
    chunk_start = start
    while chunk_start < end:
        chunk_end = min(chunk_start + pd.Timedelta(days=CHUNK_DAYS), end)
        rows.extend(_fetch_window(dataset, chunk_start, chunk_end, key))
        chunk_start = chunk_end

    if not rows:
        return pd.DataFrame({variable: []},
                            index=pd.DatetimeIndex([], tz="UTC", name="datetime"))

    frame = pd.DataFrame(rows)
    index = pd.DatetimeIndex(pd.to_datetime(frame["startTime"], utc=True, format="mixed"),
                             name="datetime")
    series = pd.Series(pd.to_numeric(frame["value"], errors="coerce").to_numpy(),
                       index=index, name=variable)
    series = series[~series.index.duplicated(keep="last")].sort_index()
    hourly = series.resample("1h").mean()
    hourly = hourly[(hourly.index >= start) & (hourly.index < end)]
    return hourly.to_frame(variable)


def load_series(variable: str, start, end) -> pd.DataFrame:
    """Cache-first :func:`fetch_series`: hit the API only on a cache miss."""
    start, end = _utc(start), _utc(end)
    path = (cache_root() /
            f"FI_{variable}_{start:%Y%m%dT%H%M}_{end:%Y%m%dT%H%M}"
            f"_1H_v{CACHE_VERSION}.parquet")
    if path.exists():
        return pd.read_parquet(path)
    if _offline():
        raise RuntimeError(
            f"SDFLOW_FINGRID_OFFLINE is set and {path} is not cached — "
            f"pre-warm the cache by loading the problem once with network access"
        )
    frame = fetch_series(variable, start, end)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    frame.to_parquet(tmp)
    tmp.replace(path)
    return frame


def fingrid_field(variable: str, start, end, *,
                  availability_lag: str = DEFAULT_AVAILABILITY_LAG) -> Field:
    """One Fingrid variable as an ``actual`` Field on an hourly UTC grid."""
    frame = load_series(variable, start, end)
    return Field(
        name=variable,
        frame=frame,
        kind="actual",
        availability_lag=availability_lag,
        description=(f"{variable} for Finland (MW, hourly), Fingrid open data "
                     f"dataset {DATASETS[variable]}"),
    )
