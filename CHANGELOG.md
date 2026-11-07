# Changelog

## 0.1.1 — 2026-11-07

- **One engine, three modes.** `DecisionEnv.step` is now four public pieces, `settle(asof)`,
  `observe(origin)`, `commit(origin, action)` and `reward(settled)`, with `pending`, `state` and
  `restore()` so a runner can drive one origin at a time. `sdflow.LiveSession(problem, model)`
  is that runner's memory: `step(dataset, now)` settles what the data makes knowable, learns
  from it, decides for the origin open now and keeps the pending decisions and state; driving
  it on truncated data replays a backtest's decisions and scores exactly, which is the test
  that backtest and live are the same loop. `Problem.sampler` (`seed -> Dataset`) and
  `sdflow.simulate(problem, make_model, n_scenarios)` run the same loop over draws of the
  exogenous information and report the score distribution; the signal game has a sampler.
  Data fetching stays outside sdflow: a live runner builds the `Dataset` as of now and hands
  it in.
- **Truncation invariance as a utility.** `sdflow.truncation_invariance(problem, make_model)`
  runs the agent on the full dataset and on datasets physically truncated at sampled origins,
  and reports every origin whose decision differs (`assert_truncation_invariant` raises). It
  catches what the feed cannot: a model holding the raw dataset, a statistic computed over
  the whole series, state carried across runs. It refuses to report a pass when a problem's
  `make_env` ignores `problem.load_dataset()`.
- **The signal game, the first non-energy benchmark.** `signal-game:w8` and `signal-game:w16`
  port the one-shot signal game from intelligence-theory: a hidden place, bit sensors and
  noise sensors, dig and be paid. Its reference scores are exact (chance, the Bayes-optimal
  reader, the oracle), so a score is a measurement against known truth and the problem doubles
  as a correctness check of the loop, the feed and the settlement. Baselines `BayesDig`,
  `LearnedDig` (fits on the training view, then learns online through `Agent.learn`) and
  `ChanceDig`.
- **Agent = Predictor ∘ Optimizer.** `Agent.act(obs)` is
  `optimizer.decide(predictor.predict(obs), obs)`: the predictor forecasts, the optimizer
  turns forecast plus state into the decision, the environment settles it. A bare predictor
  is the agent whose optimizer is the identity, so nothing that ran before changes; the same
  predictor now scores on a forecast problem and, behind an optimizer, on a decision problem.
  `Optimizer.decide` is the stage contract; `IdentityOptimizer` and `QuantileDecision` (the
  newsvendor fractile) ship. `Forecast` and `Decision` are the two frame contracts, with
  `forecast()`, `decision()`, `point_of()`, `quantile_of()` builders and accessors. The
  `Experiment` drives `act()` when a model has it and feeds every settlement to `learn()`;
  `Verifier` and `load_submission` accept agents.
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

- **A problem declares its own analyzers.** `Problem.analyzers` is a list of zero-arg
  factories; every evaluation with the default analyzers also runs them, so a score on
  the problem always carries the breakdown the problem considers meaningful. Not part
  of the spec a platform fingerprints.
- **`RegimeBreakdown`**: score, hours, share and bias (actual minus the median or point
  prediction) per regime, where the regime is read post hoc from one field's column
  against a threshold or bin edges. The hours that matter get their own number.

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
