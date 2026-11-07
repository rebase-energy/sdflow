# Changelog

## Unreleased

- **Stateful decision problems.** `DecisionEnv` is the sequential decision loop itself:
  observations carry the current `state` and the decisions still `pending` settlement; a
  `Simulator` is the transition `S_{t+1} = S^M(S_t, a_t, W_{t+1})`, applied when a decision's
  window becomes knowable; a `Settlement` turns the settled decision into its contribution.
  `ForecastEnv` is the stateless case (a `ScoreSettlement` scoring the forecast with the
  objective) and `TradingEnv` is `ForecastEnv` with a `MarketSettlement` (HEFTCom24's two-price
  rule); both keep their constructors, and either accepts `settlement=`, `action_spec=` and
  `simulator=` overrides. Action validation moved into `ActionSpec` (`ForecastAction`,
  `ColumnsAction`). `StorageSimulator` is the first simulator: a bounded store (battery,
  reservoir, warehouse) with rate limits and efficiencies that reports the flows it actually
  executed. `Experiment` runs stateful environments in event mode only, since precomputed
  actions cannot see state. `SettlementRecord.action` is the submitted action (`prediction`
  remains as an alias) and `actuals` may be `None` for problems without a target series.
- **Pre-rename settings keep working.** Every `SDFLOW_*` variable falls back to its
  `EMFLOW_*` spelling, and a cache warmed under `~/.cache/emflow/<component>` is used as long
  as no `~/.cache/sdflow/<component>` exists (`sdflow.utils.env`). Shell profiles, CI jobs and
  sandbox images configured for emflow need no change.

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
