"""Day-ahead Finnish wind power forecasting on Fingrid data with NWP predictors.

``fingrid:wind`` — at 09:00 UTC each day, forecast every hour of the next
delivery day (00:00-23:00 UTC, 24 points). Horizons run 15 h to 38 h ahead,
which is the Nord Pool day-ahead gate and far enough out that persistence is
worthless: skill has to come from the weather.

Three things make this a real problem rather than a curve fit:

* **The fleet quadrupled.** Finland went from ~2.3 GW of wind in 2022 to
  ~9.4 GW in 2026, so raw MW is not stationary across the training window and
  a model that ignores capacity will systematically under-forecast the
  present. The capacity series Fingrid uses in its own forecast is provided.
* **Winter is a different regime.** Wet snow near 0 degC ices rotors and can
  take a large share of the fleet offline while the wind is still blowing —
  the single largest source of Finnish forecast error, and invisible to a
  pure wind-speed-to-power map.
* **There is a public number to beat.** Fingrid publishes its own once-daily
  wind forecast; :data:`REFERENCE` records how well it scores on these exact
  splits, so a candidate's MAE has an operational meaning rather than only a
  relative one.

Data sources: Fingrid open data (:mod:`sdflow.data.io.fingrid`) for the
target and capacity, the rebase weather API archive
(:mod:`sdflow.data.io.weather`) for NWP. Both cache to parquet — warm them
once with network access and every later evaluation runs offline.
"""

from __future__ import annotations


import pandas as pd

import sdflow as sd
from sdflow.data import DataFeed
from sdflow.data.io.fingrid import fingrid_field
from sdflow.data.io.weather import WIND_VARIABLES, weather_field
from sdflow.envs import ForecastEnv

from .points import FI_POINTS
from sdflow.utils.env import getenv

#: The NWP archive starts here — DWD_ICON-EU is not served before 2022, and
#: it, not Fingrid, is what bounds the problem's history.
DATA_START = pd.Timestamp(getenv("FINGRID_DATA_START",
                                         "2022-01-01T00:00:00Z"))
DATA_END = pd.Timestamp(getenv("FINGRID_DATA_END",
                                       "2026-08-20T00:00:00Z"))

TARGET_FIELD = "wind"
CAPACITY_FIELD = "wind_capacity"
NWP_FIELD = "nwp"

#: Measured generation is knowable an hour after the fact; installed capacity
#: is a slow-moving stock that is knowable as it stands (nobody learns
#: tomorrow's capacity tomorrow — it is already built).
TARGET_LAG = "1h"
CAPACITY_LAG = "0h"

NWP_MODEL = getenv("FINGRID_NWP_MODEL", "DWD_ICON-EU")

#: One origin per day, so splits are measured in origins: 120 days = 120
#: origins = 2880 scored hours.
HOLDOUT_DAYS = 120
VALIDATION_DAYS = 120

#: 09:00 UTC issue; targets 00:00-23:00 UTC the next day (15 h .. 38 h ahead).
ORIGIN_TIME = "09:00"
COVER_START = "15h"
COVER_END = "38h"

SPLITS = sd.Splits(
    train_end=DATA_END - pd.Timedelta(days=HOLDOUT_DAYS + VALIDATION_DAYS),
    validation=(DATA_END - pd.Timedelta(days=HOLDOUT_DAYS + VALIDATION_DAYS),
                DATA_END - pd.Timedelta(days=HOLDOUT_DAYS)),
    holdout=(DATA_END - pd.Timedelta(days=HOLDOUT_DAYS), DATA_END),
)

#: Fingrid's own once-daily wind forecast (dataset 246) scored against
#: measured generation on these exact splits, MAE in MW. Measured 2026-08-21.
#:
#: Read this as an indicative bar rather than an exact like-for-like: the API
#: serves that forecast by valid time with no issue time, so some values may
#: be later, better-informed updates than a true day-ahead lead. If anything
#: it flatters Fingrid, which makes it the honest direction to err in.
#: For scale, predicting the trailing mean capacity factor every hour scores
#: 1374 MW on the holdout, so the NWP is worth roughly 1000 MW of MAE.
REFERENCE: dict = {"validation": 563.5, "holdout": 360.0}


def nwp_columns() -> list[str]:
    """Every NWP column: ``<point>__<variable>``, with ``:`` folded to ``_``."""
    return [f"{p.name}__{v.replace(':', '_')}"
            for p in FI_POINTS for v in WIND_VARIABLES]


def _reference_line() -> str:
    if not REFERENCE:
        return ("Fingrid's own once-daily wind forecast is public (dataset "
                "246); its MAE on these splits is not recorded here yet.")
    parts = ", ".join(f"{split} {mae:.1f} MW" for split, mae in REFERENCE.items())
    return (f"Reference: Fingrid's own published wind forecast scores {parts} "
            f"MAE on these exact splits — the operational bar, and an "
            f"indicative one (its issue times are not published, so it may "
            f"hold a lead-time advantage). Predicting the trailing mean "
            f"capacity factor every hour scores 1374 MW on the holdout, so "
            f"nearly all the available skill comes from the weather.")


def _description() -> str:
    return (
        f"Day-ahead forecasting of Finnish wind power generation (MW, hourly), "
        f"on Fingrid open data (dataset 75), {DATA_START.date()} to "
        f"{DATA_END.date()}.\n\n"
        f"At 09:00 UTC each day, forecast all 24 hours of the next delivery "
        f"day (00:00-23:00 UTC) — horizons 15h to 38h. This is the Nord Pool "
        f"day-ahead gate. Scored by MAE (MW).\n\n"
        f"Target field: {TARGET_FIELD!r} (single column, hourly). Actuals "
        f"become knowable {TARGET_LAG} after their timestamp, so at the 09:00 "
        f"origin the newest actual is from 08:00 that morning — 16 hours "
        f"before the first target. Persistence is not a viable strategy at "
        f"this range.\n\n"
        f"Predictor field {NWP_FIELD!r}: archived {NWP_MODEL} runs at "
        f"{len(FI_POINTS)} locations across Finland's wind build-out "
        f"({', '.join(p.name for p in FI_POINTS)}), variables "
        f"{', '.join(WIND_VARIABLES)}. It is a forecast field indexed by "
        f"(issue_time, valid_time): the feed serves the newest run already "
        f"disseminated at the origin, which at 09:00 UTC is that day's 00Z "
        f"run. Read it with ForecastField({NWP_FIELD!r}, columns=(...)).\n\n"
        f"Predictor field {CAPACITY_FIELD!r}: installed wind capacity (MW, "
        f"the series Fingrid uses in its own forecast). Finland's fleet grew "
        f"from ~2.3 GW to ~9.4 GW over this window, so the same wind means "
        f"very different MW in 2022 and 2026 — modelling the capacity factor "
        f"rather than raw MW is usually the single biggest win here.\n\n"
        f"Physics worth knowing: power responds to hub-height wind speed "
        f"through a sigmoid curve that cuts in near 3 m/s, saturates at rated "
        f"speed around 12-13 m/s and cuts out in storms; output scales with "
        f"air density, so cold air yields more power at the same speed; and "
        f"in winter, wet snow near 0 degC ices rotors and can idle a large "
        f"part of the fleet while the wind still blows.\n\n"
        f"{_reference_line()}\n\n"
        f"Splits: train < {SPLITS.train_end.date()}, validation "
        f"{SPLITS.validation[0].date()} .. {SPLITS.validation[1].date()}, "
        f"holdout {SPLITS.holdout[0].date()} .. {SPLITS.holdout[1].date()}."
    )


def _load_dataset() -> sd.Dataset:
    from sdflow.problems.registry import ProblemNotIngestedError

    target = fingrid_field(TARGET_FIELD, DATA_START, DATA_END,
                           availability_lag=TARGET_LAG)
    first_valid = target.frame[TARGET_FIELD].first_valid_index()
    if first_valid is None or first_valid > SPLITS.train_end - pd.Timedelta(days=180):
        raise ProblemNotIngestedError(
            f"fingrid:wind: Fingrid has generation only from {first_valid} — "
            f"not enough history before train_end {SPLITS.train_end}"
        )
    capacity = fingrid_field(CAPACITY_FIELD, DATA_START, DATA_END,
                             availability_lag=CAPACITY_LAG)
    # Installed capacity is a stock, not a flow: a published 0 MW or a gap is
    # a glitch, never a fleet that vanished for an hour (11 such points in
    # 4.6 years). Carry the last real value forward, and back-fill the very
    # first rows. Left raw, a candidate dividing by capacity gets an inf.
    frame = capacity.frame.replace({CAPACITY_FIELD: {0.0: None}})
    capacity.frame = frame.ffill().bfill()
    # NWP runs must reach COVER_END past the last target, or the final origins
    # have no weather to forecast with.
    nwp = weather_field(NWP_FIELD, FI_POINTS, NWP_MODEL,
                        DATA_START, DATA_END + pd.Timedelta(COVER_END))
    return sd.Dataset(
        name="fingrid-wind",
        description=("Finnish wind generation and capacity (Fingrid) with "
                     f"{NWP_MODEL} forecasts at {len(FI_POINTS)} points "
                     "(rebase weather archive)"),
        fields={TARGET_FIELD: target, CAPACITY_FIELD: capacity, NWP_FIELD: nwp},
    )


def make_env(problem: sd.Problem, split: str) -> ForecastEnv:
    return ForecastEnv(
        feed=DataFeed(problem.load_dataset()),
        origins=problem.origins(split),
        target_field=TARGET_FIELD,
        objective=problem.objective,
        train_end=problem.splits.train_end,
    )


def get_problem() -> sd.Problem:
    return sd.Problem(
        name="fingrid:wind",
        dataset=_load_dataset,
        make_env=make_env,
        objective=sd.Objective(sd.MeanAbsoluteError()),
        schedule=sd.IssueSchedule(origin_freq="1D", cover_start=COVER_START,
                                  cover_end=COVER_END, target_freq="1h",
                                  origin_time=ORIGIN_TIME),
        splits=SPLITS,
        description=_description(),
    )


def list_problem_variants():
    return ["wind"]


get_problem_wind = get_problem
