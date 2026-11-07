<div align="center">
<h1>sdflow</h1>
<p><b>Open-source Python framework for sequential decision problems.</b></p>
</div>

A sequential decision problem is a loop. State **S_t** is observed, a decision
**a_t = A^π(S_t)** is taken, exogenous information **W_{t+1}** is revealed, a new state
**S_{t+1} = S^M(S_t, a_t, W_{t+1})** is reached and a contribution **C(S_t, a_t, W_{t+1})**
is earned. The goal is a policy **π** that maximises the expected total contribution.

sdflow gives that loop a concrete form:

- **Problems** built from point-in-time **datasets**, **environments** and **objectives**:
  every field is bitemporal, so a model physically cannot read data that was not knowable
  at the decision origin.
- **Policies** built from **predictors**, **optimizers** and **simulators**, connected by
  **agents**.
- **One engine** that runs a policy the same way in a backtest over history, a simulation
  over synthetic exogenous information, and live operation.
- **Verifiable scoring**: official splits, a `Verifier` that fits a fresh submission itself
  and scores the hidden holdout, bundled benchmarks with reference baselines.

The core is domain-neutral. Energy, the best-supported domain, is an extra:
`sdflow[energy]` adds the asset classes built on
[energydatamodel](https://github.com/rebase-energy/energydatamodel). sdflow is the
successor of [emflow](https://github.com/rebase-energy/emflow); emflow 0.3 code ports by
renaming the import.

## Installation

```bash
pip install sdflow            # the core
pip install "sdflow[energy]"  # plus the energy asset classes
```

## Quick start

```python
import sdflow as sd

problem = sd.load_problem("gefcom2014:solar")      # a bundled benchmark
result = sd.evaluate(problem, sd.Predictor.persistence(), split="validation")
print(result.score, result.n_origins)
```

A predictor is `fit(train)` on a leakage-safe training view and `predict(obs)` per decision
origin; the environment computes the objective from what it returns. See `docs/` for the
walkthrough: datasets and feeds, environments, features, models, problems and the run engine.

## Agents

A policy is a predictor and an optimizer. The predictor forecasts, the optimizer turns the
forecast and the state into the decision, the environment settles it:

```python
from sdflow.models import Agent, QuantileDecision

agent = Agent(my_quantile_predictor, QuantileDecision(c_under=3, c_over=1))
result = sd.evaluate("heftcom2024:trading", agent)        # revenue, not pinball loss
```

A predictor on its own is the agent whose optimizer is the identity, so the same predictor
can be scored on a forecast leaderboard and, behind an optimizer, on a revenue one.

## Stateful problems

A forecast problem is the stateless case of the loop. When a decision changes what the next
decision faces, give the environment a simulator and a settlement:

```python
from sdflow.envs import ColumnsAction, DecisionEnv, Settlement
from sdflow.models import StorageSimulator

class StorageRevenue(Settlement):
    fields = ("price",)                      # settle once the realised price is knowable
    def settle(self, ctx):                   # price what the store actually did
        return float(-(ctx.series("price") * ctx.state_after.realized).sum())

env = DecisionEnv(
    feed, origins, settlement=StorageRevenue(), objective=objective,
    action_spec=ColumnsAction("flow"),       # the decision: a flow per target period
    simulator=StorageSimulator(capacity=10, max_inflow=5, max_outflow=5),
)
obs, _ = env.reset()                         # obs.state.level == 0.0, obs.pending == ()
```

The observation carries the state after the last settled decision and the decisions still
pending, which is exactly what a live participant knows at that clock time: committed, not
yet metered. `ForecastEnv` and `TradingEnv` are this environment with a scoring and a
market settlement.

## Leak checks

The feed makes look-ahead hard by construction. For what construction cannot catch, run the
truncation check: the decision at an origin must be identical when everything not knowable at
that origin is physically deleted from the dataset.

```python
report = sd.truncation_invariance(problem, lambda: MyAgent(seed=0), k=6)
assert report.ok, report
```

`signal-game:w8` is the framework's own check: a problem with exact reference scores (chance,
Bayes, oracle) and no energy in it.

## Registering your own problems

Problems in other packages register through the `sdflow.problems` entry-point group and
then load by name anywhere the package is installed:

```toml
[project.entry-points."sdflow.problems"]
icing = "my_signals.icing.problem"
```

```python
sd.load_problem("icing:se1")
```

Data can live on the Hugging Face Hub: a dataset repo with a `rebase.yaml` marker is
readable through the `rb://` filesystem sdflow registers with fsspec
(`pd.read_parquet("rb://dataset/<owner>/<repo>/<path>")`).

## Development

```bash
uv sync --group dev
uv run pytest
```

MIT licensed. Built by [rebase.energy](https://www.rebase.energy).
