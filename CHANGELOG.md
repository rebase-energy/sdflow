# Changelog

## 0.1.0 — 2026-11-07

First release of sdflow: the domain-neutral port of emflow 0.3.1.

- Everything emflow had, under the `sdflow` name: datasets, fields and point-in-time
  feeds; `ForecastEnv` and `TradingEnv`; features; predictors, optimizers, simulators and
  agents; problems, schedules, splits and metrics; the run engine with `evaluate`,
  `Experiment`, `Verifier`, analyzers and submissions; bundled benchmarks (GEFCom 2012/2014/
  2017, BFCom 2018, BigDEAL 2022, HEFTCom 2024, Fingrid wind, grid) and the Swedish
  temperatures example; the Hugging Face-backed `rb://` filesystem.
- **Energy is an extra.** The core imports nothing energy-specific. The asset classes built
  on energydatamodel moved to `sdflow.energy` (`pip install "sdflow[energy]"`);
  `Dataset.collection` and `BaseProblem.collection` take any object.
- **External problems** register through the `sdflow.problems` entry-point group (was
  `emflow.problems`).
- Dropped the `_legacy` example notebooks.

The emflow changelog this continues from is kept as `CHANGELOG.emflow.md`.
