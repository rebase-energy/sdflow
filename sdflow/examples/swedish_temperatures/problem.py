"""Swedish temperature forecasting problem (Stockholm, 1-step-ahead AR).

Hourly air temperature from the SMHI station Stockholm-Observatoriekullen A
(id ``98230``). The task: train on everything before 2026 and forecast one hour
ahead, origin by origin, through 2026. Loaded via
``sdflow.load_problem("swedish-temperatures:ar")``.
"""

import importlib.resources

import pandas as pd

import sdflow as sd
from sdflow.data import DataFeed
from sdflow.envs import ForecastEnv

STATION_ID = "98230"
STATION_NAME = "Stockholm-Observatoriekullen A"

# Station data is complete through Feb 2026 (the SMHI feed for 98230 stops
# 2026-03-01); agents iterate on January, the verifier scores February.
SPLITS = sd.Splits(
    train_end="2026-01-01T00:00:00Z",
    validation=("2026-01-01T00:00:00Z", "2026-02-01T00:00:00Z"),
    holdout=("2026-02-01T00:00:00Z", "2026-03-01T00:00:00Z"),
)


def load_dataset() -> sd.Dataset:
    with importlib.resources.path("sdflow.examples.data", "swedish-temperatures.parquet") as path:
        df = pd.read_parquet(path)
    with importlib.resources.path("sdflow.examples.data", "swedish-temperatures-stations.parquet") as path:
        meta = pd.read_parquet(path)

    temps = df.xs(STATION_ID, level="station_id", axis=1)
    temps.columns = ["temperature"]

    # The entity the data describes. Nothing energy-specific here, so a plain record
    # rather than an energydatamodel collection (see sdflow.energy for those).
    portfolio = {"name": "SwedishStations", "members": [{"name": STATION_NAME, "id": STATION_ID}]}

    return sd.Dataset(
        name="swedish-temperatures",
        description=(
            "Hourly air temperature (°C) from SMHI station "
            f"{STATION_NAME} (id {STATION_ID}). Benchmark for 1-step-ahead "
            "autoregressive forecasting: train < 2026, evaluate on 2026."
        ),
        collection=portfolio,
        fields={
            "temperature": sd.Field("temperature", temps, availability_lag="0h"),
            "stations": sd.Field.static("stations", meta),
        },
    )


def make_env(problem: sd.Problem, split: str) -> ForecastEnv:
    return ForecastEnv(
        feed=DataFeed(problem.load_dataset()),
        origins=problem.origins(split),
        target_field="temperature",
        objective=problem.objective,
        train_end=problem.splits.train_end,
    )


def list_problem_variants():
    return ["ar"]


def get_problem_ar() -> sd.Problem:
    return sd.Problem(
        name="swedish-temperatures:ar",
        dataset=load_dataset,
        make_env=make_env,
        objective=sd.Objective(sd.MeanAbsoluteError()),
        schedule=sd.IssueSchedule.hourly(horizon="1h"),
        splits=SPLITS,
        description=__doc__,
    )
