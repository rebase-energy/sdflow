"""Problems declared in a repo's rebase.yaml load and run with no Python package."""

import numpy as np
import pandas as pd
import pytest
import yaml

import sdflow as sd
from sdflow.models import Predictor
from sdflow.models.frames import forecast
from sdflow.problems import ManifestError, manifest_problems
from sdflow.problems.manifest import split_reference, is_manifest_reference


def _stage(root, *, problems):
    """A two-farm repo: hourly power and a daily 00 UTC forecast per farm."""
    index = pd.date_range("2024-01-01", "2024-04-01", freq="1h", tz="UTC", inclusive="left")
    (root / "a").mkdir()
    (root / "b").mkdir()
    rng = np.random.default_rng(0)
    for farm, level in (("a", 10.0), ("b", 20.0)):
        power = pd.DataFrame({"power": level + rng.normal(0, 1, len(index))}, index=index)
        power.index.name = "time"
        power.to_parquet(root / farm / "power.parquet")
        issue = index[index.hour == 0]
        rows = [(i, i + pd.Timedelta(hours=lead), level) for i in issue for lead in range(12, 36)]
        nwp = pd.DataFrame(rows, columns=["issue_time", "valid_time", "ws"])
        nwp.to_parquet(root / farm / "nwp.parquet", index=False)
    manifest = {
        "name": "toy",
        "description": "two farms",
        "fields": {
            f"{farm}/{field}": {
                "path": f"{farm}/{field}.parquet",
                "kind": "forecast" if field == "nwp" else "actual",
                "availability_lag": "3h" if field == "nwp" else "2h",
                **({"issue_col": "issue_time", "valid_col": "valid_time"} if field == "nwp" else {}),
            }
            for farm in ("a", "b")
            for field in ("power", "nwp")
        },
        "problems": problems,
    }
    (root / "rebase.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
    return root


def _problem(farm, **overrides):
    spec = {
        "description": f"farm {farm}",
        "fields": {"power": f"{farm}/power", "nwp": f"{farm}/nwp"},
        "target": {"field": "power", "column": "power"},
        "objective": {"metric": "mae"},
        "schedule": {"daily": {"at": "10:00", "covers": ["12h", "35h"], "target_freq": "1h"}},
        "splits": {
            "train_end": "2024-02-01T00:00:00Z",
            "validation": ["2024-02-01T00:00:00Z", "2024-03-01T00:00:00Z"],
            "holdout": ["2024-03-01T00:00:00Z", "2024-04-01T00:00:00Z"],
        },
        "reference_scores": [{"team": "climatology", "score": 0.8}],
    }
    spec.update(overrides)
    return spec


class TrainMean(Predictor):
    def fit(self, train):
        self.mean = float(train.history("power")["power"].mean())
        return self

    def predict(self, obs):
        return forecast(obs.target_index, point=self.mean)


@pytest.fixture
def repo(tmp_path):
    return _stage(tmp_path, problems={"a": _problem("a"), "b": _problem("b")})


class TestReference:
    def test_registered_names_are_not_references(self):
        assert not is_manifest_reference("gefcom2014:solar")
        assert is_manifest_reference("rb://dataset/o/n:x")
        assert is_manifest_reference("o/n")
        assert is_manifest_reference("/tmp/staged:x")

    def test_variant_split_leaves_uri_colons_alone(self):
        assert split_reference("rb://dataset/o/n:odal") == ("rb://dataset/o/n", "odal")
        assert split_reference("rb://dataset/o/n") == ("rb://dataset/o/n", None)
        assert split_reference("/tmp/staged:odal") == ("/tmp/staged", "odal")
        assert split_reference("o/n") == ("o/n", None)


class TestLoad:
    def test_lists_and_loads_each_problem(self, repo):
        assert manifest_problems(str(repo)) == ["toy:a", "toy:b"]
        problem = sd.load_problem(f"{repo}:b")
        assert problem.name == "toy:b"
        assert problem.description == "farm b"
        assert problem.objective.name == "MeanAbsoluteError"
        assert problem.splits.train_end == pd.Timestamp("2024-02-01", tz="UTC")
        assert [r.team for r in problem.reference_scores] == ["climatology"]
        assert problem.reference_scores[0].rank == 1

    def test_only_the_problems_fields_are_read(self, repo):
        dataset = sd.load_problem(f"{repo}:a").load_dataset()
        assert sorted(dataset.fields) == ["nwp", "power"]
        assert dataset.field("nwp").kind == "forecast"
        assert float(dataset.field("power").frame["power"].mean()) == pytest.approx(10.0, abs=0.2)

    def test_evaluates_end_to_end(self, repo):
        result = sd.evaluate(f"{repo}:b", TrainMean(), split="validation")
        assert result.n_origins == 30            # 29 days of February 2024 plus the origin straddling the 1st
        assert 0.5 < result.score < 1.5          # MAE of the mean on unit noise

    def test_list_fields_name_by_basename(self, tmp_path):
        repo = _stage(tmp_path, problems={"a": _problem("a", fields=["a/power", "a/nwp"])})
        dataset = sd.load_problem(f"{repo}:a").load_dataset()
        assert sorted(dataset.fields) == ["nwp", "power"]

    def test_single_problem_needs_no_variant(self, tmp_path):
        repo = _stage(tmp_path, problems={"only": _problem("a")})
        assert sd.load_problem(str(repo)).name == "toy:only"

    def test_dataset_from_manifest_reads_a_directory(self, repo):
        dataset = sd.Dataset.from_manifest(str(repo))
        assert sorted(dataset.fields) == ["a/nwp", "a/power", "b/nwp", "b/power"]


class TestErrors:
    def test_unknown_variant(self, repo):
        with pytest.raises(ManifestError, match="no problem 'c'"):
            sd.load_problem(f"{repo}:c")

    def test_ambiguous_without_variant(self, repo):
        with pytest.raises(ManifestError, match="declares 2 problems"):
            sd.load_problem(str(repo))

    def test_unknown_field_key(self, tmp_path):
        repo = _stage(tmp_path, problems={"a": _problem("a", fields={"power": "a/power", "x": "a/missing"})})
        with pytest.raises(ManifestError, match="a/missing"):
            sd.load_problem(f"{repo}:a")

    def test_unknown_metric(self, tmp_path):
        repo = _stage(tmp_path, problems={"a": _problem("a", objective={"metric": "crps"})})
        with pytest.raises(ManifestError, match="objective.metric"):
            sd.load_problem(f"{repo}:a")

    def test_pinball_needs_quantiles(self, tmp_path):
        repo = _stage(tmp_path, problems={"a": _problem("a", objective={"metric": "pinball"})})
        with pytest.raises(ManifestError, match="quantiles"):
            sd.load_problem(f"{repo}:a")

    def test_pinball_sets_env_quantiles(self, tmp_path):
        spec = _problem("a", objective={"metric": "pinball", "quantiles": [0.1, 0.5, 0.9]})
        repo = _stage(tmp_path, problems={"a": spec})
        env = sd.load_problem(f"{repo}:a").env("validation")
        assert tuple(env.quantiles) == (0.1, 0.5, 0.9)

    def test_no_problems_section(self, tmp_path):
        repo = _stage(tmp_path, problems={})
        with pytest.raises(ManifestError, match="declares no problems"):
            sd.load_problem(str(repo))


class TestStraddlingOrigin:
    def test_fit_once_model_is_truncation_invariant(self, repo):
        """The first validation origin is issued at 10:00 the day before the split starts.
        The training view stops at that origin, so a model fitted once sees nothing the
        origin cannot know and the leak check passes."""
        problem = sd.load_problem(f"{repo}:a")
        env = problem.env("validation")
        assert env.origins[0].asof < problem.splits.train_end
        assert env.train_asof == env.origins[0].asof
        report = sd.truncation_invariance(problem, TrainMean, split="validation", k=3)
        assert report.ok, report.violations
