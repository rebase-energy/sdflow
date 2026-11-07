"""The rename must not orphan EMFLOW_* variables or ~/.cache/emflow caches."""

from pathlib import Path

from sdflow.utils import env as env_utils


def test_getenv_prefers_new_name_then_legacy_then_default(monkeypatch):
    monkeypatch.delenv("SDFLOW_X", raising=False)
    monkeypatch.delenv("EMFLOW_X", raising=False)
    assert env_utils.getenv("X", "dflt") == "dflt"
    monkeypatch.setenv("EMFLOW_X", "old")
    assert env_utils.getenv("X") == "old"
    monkeypatch.setenv("SDFLOW_X", "new")
    assert env_utils.getenv("X") == "new"


def test_getenv_treats_empty_as_unset(monkeypatch):
    monkeypatch.setenv("SDFLOW_X", "")
    monkeypatch.setenv("EMFLOW_X", "old")
    assert env_utils.getenv("X") == "old"


def test_cache_root_uses_legacy_dir_only_when_new_one_is_absent(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(env_utils, "CACHE_HOME", tmp_path)
    monkeypatch.delenv("SDFLOW_GRID_CACHE", raising=False)
    monkeypatch.delenv("EMFLOW_GRID_CACHE", raising=False)
    new, old = tmp_path / "sdflow" / "grid", tmp_path / "emflow" / "grid"
    assert env_utils.cache_root("grid") == new          # neither exists: the new path
    old.mkdir(parents=True)
    assert env_utils.cache_root("grid") == old          # only the warmed legacy cache exists
    new.mkdir(parents=True)
    assert env_utils.cache_root("grid") == new          # both exist: the new one wins


def test_cache_root_env_override_accepts_legacy_variable(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("SDFLOW_GRID_CACHE", raising=False)
    monkeypatch.setenv("EMFLOW_GRID_CACHE", str(tmp_path / "elsewhere"))
    assert env_utils.cache_root("grid") == tmp_path / "elsewhere"
