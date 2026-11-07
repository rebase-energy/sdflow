"""Weather loader and the fingrid:wind problem — offline (seeded caches)."""

import numpy as np
import pandas as pd
import pytest

from sdflow.data.field import ISSUE_LEVEL, VALID_LEVEL
from sdflow.data.io import weather
from sdflow.data.io.weather import Point


def _payload(issue_times, horizon=48, variables=("WindSpeed:100",)):
    """A response shaped like /weather/v2/point/historical: parallel arrays."""
    ref, valid = [], []
    for issue in issue_times:
        issue = pd.Timestamp(issue)
        for h in range(horizon):
            ref.append(issue.isoformat())
            valid.append((issue + pd.Timedelta(hours=h)).isoformat())
    record = {"ref_datetime": ref, "valid_datetime": valid}
    for i, name in enumerate(variables):
        record[name] = [5.0 + i + (j % 7) for j in range(len(ref))]
    return [record]


def _seed_cache(tmp_path, monkeypatch, point, model, start, end, *,
                variables=weather.WIND_VARIABLES):
    monkeypatch.setenv("SDFLOW_WEATHER_CACHE", str(tmp_path))
    start, end = weather._utc(start), weather._utc(end)
    issues = pd.date_range(start, end, freq="1D", inclusive="left", tz="UTC")
    frame = weather._runs_frame(_payload(issues, variables=variables), point, variables)
    valid = frame.index.get_level_values(VALID_LEVEL)
    frame = frame[(valid >= start) & (valid < end)]
    stem = (f"{point.name}_{point.latitude:.3f}_{point.longitude:.3f}_{model}_"
            f"{start:%Y%m%dT%H%M}_{end:%Y%m%dT%H%M}_24H_full_{len(variables)}v"
            f"_v{weather.CACHE_VERSION}")
    frame.to_parquet(tmp_path / f"{stem}.parquet")
    return frame


POINT = Point("vaasa", 63.10, 21.60, weight=1.0)


class TestRunsFrame:
    def test_parses_parallel_arrays_into_issue_valid_index(self):
        frame = weather._runs_frame(
            _payload(["2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z"], horizon=3),
            POINT, ("WindSpeed:100",),
        )
        assert frame.index.names == [ISSUE_LEVEL, VALID_LEVEL]
        assert list(frame.columns) == ["vaasa__WindSpeed_100"]
        assert len(frame) == 6
        issues = frame.index.get_level_values(ISSUE_LEVEL).unique()
        assert len(issues) == 2 and issues.tz is not None

    def test_missing_variable_becomes_a_nan_column_not_an_error(self):
        payload = _payload(["2026-01-01T00:00:00Z"], horizon=2,
                           variables=("WindSpeed:100",))
        frame = weather._runs_frame(payload, POINT, ("WindSpeed:100", "AirDensity"))
        assert list(frame.columns) == ["vaasa__WindSpeed_100", "vaasa__AirDensity"]
        assert frame["vaasa__AirDensity"].isna().all()

    def test_response_without_time_columns_is_a_clear_error(self):
        with pytest.raises(RuntimeError, match="lacks"):
            weather._runs_frame([{"WindSpeed:100": [1.0]}], POINT, ("WindSpeed:100",))


class TestLoadPointRuns:
    def test_cache_hit_needs_no_network_or_key(self, monkeypatch, tmp_path):
        monkeypatch.delenv("SDFLOW_WEATHER_API_KEY", raising=False)
        monkeypatch.delenv("REBASE_API_KEY", raising=False)
        seeded = _seed_cache(tmp_path, monkeypatch, POINT, "DWD_ICON-EU",
                             "2026-01-01", "2026-01-11")
        monkeypatch.setenv("SDFLOW_WEATHER_OFFLINE", "1")
        out = weather.load_point_runs(POINT, "DWD_ICON-EU", "2026-01-01", "2026-01-11")
        pd.testing.assert_frame_equal(out, seeded)

    def test_offline_cache_miss_is_a_clear_error(self, monkeypatch, tmp_path):
        monkeypatch.setenv("SDFLOW_WEATHER_CACHE", str(tmp_path))
        monkeypatch.setenv("SDFLOW_WEATHER_OFFLINE", "1")
        with pytest.raises(RuntimeError, match="SDFLOW_WEATHER_OFFLINE"):
            weather.load_point_runs(POINT, "DWD_ICON-EU", "2026-01-01", "2026-02-01")

    def test_fetch_without_key_is_a_clear_error(self, monkeypatch, tmp_path):
        monkeypatch.setenv("SDFLOW_WEATHER_CACHE", str(tmp_path))
        monkeypatch.delenv("SDFLOW_WEATHER_OFFLINE", raising=False)
        monkeypatch.delenv("SDFLOW_WEATHER_API_KEY", raising=False)
        monkeypatch.delenv("REBASE_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="no API key"):
            weather.load_point_runs(POINT, "DWD_ICON-EU", "2026-01-01", "2026-02-01")


class TestWeatherField:
    def test_joins_points_into_one_forecast_field(self, monkeypatch, tmp_path):
        other = Point("satakunta", 61.40, 21.50, weight=1.0)
        for point in (POINT, other):
            _seed_cache(tmp_path, monkeypatch, point, "DWD_ICON-EU",
                        "2026-01-01", "2026-01-11")
        monkeypatch.setenv("SDFLOW_WEATHER_OFFLINE", "1")
        field = weather.weather_field("nwp", [POINT, other], "DWD_ICON-EU",
                                      "2026-01-01", "2026-01-11")
        assert field.kind == "forecast"
        assert len(field.frame.columns) == 2 * len(weather.WIND_VARIABLES)
        assert field.frame.columns[0].startswith("vaasa__")
        assert any(c.startswith("satakunta__") for c in field.frame.columns)

    def test_availability_lag_defaults_to_the_models_dissemination_delay(
            self, monkeypatch, tmp_path):
        _seed_cache(tmp_path, monkeypatch, POINT, "DWD_ICON-EU",
                    "2026-01-01", "2026-01-11")
        monkeypatch.setenv("SDFLOW_WEATHER_OFFLINE", "1")
        field = weather.weather_field("nwp", [POINT], "DWD_ICON-EU",
                                      "2026-01-01", "2026-01-11")
        assert field.availability_lag == pd.Timedelta("4h")

    def test_no_points_is_a_clear_error(self):
        with pytest.raises(ValueError, match="no points"):
            weather.weather_field("nwp", [], "DWD_ICON-EU", "2026-01-01", "2026-01-11")


class TestProblemFamily:
    def test_registered_under_fingrid(self):
        from sdflow.problems.registry import list_problems

        assert "fingrid:wind" in list_problems()

    def test_day_ahead_origins_cover_the_next_delivery_day(self):
        from sdflow.benchmarks.fingrid import problem as fingrid_problem

        problem = fingrid_problem.get_problem()
        origins = problem.origins("validation")
        assert len(origins) == fingrid_problem.VALIDATION_DAYS
        for origin in origins[:5]:
            assert origin.asof.strftime("%H:%M") == "09:00"
            assert len(origin.target_index) == 24
            assert origin.target_start == (origin.asof.normalize()
                                           + pd.Timedelta(days=1))
            assert origin.target_end == origin.target_start + pd.Timedelta(hours=23)

    def test_splits_do_not_overlap(self):
        from sdflow.benchmarks.fingrid import problem as fingrid_problem

        assert fingrid_problem.SPLITS.train_end == fingrid_problem.SPLITS.validation[0]
        assert fingrid_problem.SPLITS.validation[1] == fingrid_problem.SPLITS.holdout[0]

    def test_nwp_columns_match_what_the_loader_emits(self):
        from sdflow.benchmarks.fingrid import problem as fingrid_problem
        from sdflow.benchmarks.fingrid.points import FI_POINTS

        columns = fingrid_problem.nwp_columns()
        assert "vaasa__WindSpeed_100" in columns
        assert len(columns) == len(FI_POINTS) * len(weather.WIND_VARIABLES)
        assert all(":" not in c for c in columns)


class TestPowerCurveBaseline:
    def test_scales_the_capacity_factor_curve_by_capacity(self):
        from sdflow.benchmarks.fingrid.baseline import PowerCurve, SPEED_COLUMNS
        from sdflow.benchmarks.fingrid.problem import CAPACITY_FIELD

        model = PowerCurve()
        model._speeds = np.array([3.0, 8.0, 15.0])
        model._capacity_factor = np.array([0.0, 0.35, 0.90])
        columns = list(SPEED_COLUMNS) + [f"{CAPACITY_FIELD}_lag_24h"]
        X = pd.DataFrame(
            [[8.0] * len(SPEED_COLUMNS) + [10_000.0],   # mid curve, big fleet
             [8.0] * len(SPEED_COLUMNS) + [2_500.0],    # same wind, small fleet
             [0.0] * len(SPEED_COLUMNS) + [10_000.0]],  # below cut-in
            columns=columns)
        out = model.predict_tabular(X)["point"]
        assert out.iloc[0] == pytest.approx(3500.0)
        assert out.iloc[1] == pytest.approx(875.0)
        assert out.iloc[0] / out.iloc[1] == pytest.approx(4.0)
        assert out.iloc[2] == pytest.approx(0.0)

    def test_never_predicts_negative_power(self):
        from sdflow.benchmarks.fingrid.baseline import PowerCurve, SPEED_COLUMNS
        from sdflow.benchmarks.fingrid.problem import CAPACITY_FIELD

        model = PowerCurve()
        model._speeds = np.array([3.0, 8.0])
        model._capacity_factor = np.array([-0.05, 0.4])   # a noisy low bin
        X = pd.DataFrame([[2.0] * len(SPEED_COLUMNS) + [9000.0]],
                         columns=list(SPEED_COLUMNS) + [f"{CAPACITY_FIELD}_lag_24h"])
        assert model.predict_tabular(X)["point"].iloc[0] == 0.0
