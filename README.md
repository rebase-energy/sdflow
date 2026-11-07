# sdflow

Open-source Python framework for modelling sequential decision problems.

State information **S_t** is observed, a decision **a_t = A^π(S_t)** is taken, exogenous
information **W_{t+1}** is revealed, a new state **S_{t+1} = S^M(S_t, a_t, W_{t+1})** is reached
and a contribution **C(S_t, a_t, W_{t+1})** is earned. The loop repeats until the horizon.
The goal is a policy **π** that maximises the expected total contribution.

**sdflow** gives that loop a concrete form: problems built from datasets, environments and
objectives; policies built from predictors, optimizers and simulators, connected by agents;
and one engine that runs a policy the same way in a backtest over history, a simulation
over synthetic exogenous information, and live operation.

It is the domain-neutral successor of [emflow](https://github.com/rebase-energy/emflow),
whose energy-specific components will become an extra on top of it.

> **Status:** name reserved; the port from emflow is in progress. Nothing to use yet.

## Installation

```bash
pip install sdflow
```

## Licence

MIT, see [LICENSE](LICENSE).
