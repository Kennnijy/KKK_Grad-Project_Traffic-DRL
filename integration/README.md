# `integration/` — prediction → decision dataflow

This folder wires the traffic-prediction models (STGCN + STGAT) to the routing
decision layer. It builds a congestion-weighted road graph from the predictions and
compares routing policies, measuring whether congestion-aware + global-penalty
routing suppresses the **herding effect (羊群效應)** — the core claim of the project.

Everything needed for the decision stage lives in this one folder; it has no sideways
dependency on any other directory. The road-network adjacency is vendored under
`data/`, and the model predictions are written here by the model repos' `run_infer.py`.

## How to run

```
cd integration
python pipeline.py                 # full: (re)generate predictions on GPU, then compare
python pipeline.py --skip-infer    # reuse cached *_pred.npy (fast, no GPU)
python run_compare.py --scenario random --vehicles 200   # run the comparison directly
```

The decision stage is pure NumPy + NetworkX + SciPy (no GPU, ~1 s). Only the optional
inference stage needs PyTorch/GPU.

## Files

| file | role |
|---|---|
| `config.py` | paths + all hyper-parameters (single source of truth) |
| `network.py` | load predictions + adjacency, build the directed road graph |
| `policies.py` | routing policies (static / prediction-greedy / load-aware / global-penalty) + DRL scaffold (RoutingEnv / EGATActorCritic / PPOTrainer / policy_drl) |
| `metrics.py` | ATT, Gini(edge load), worst-link saturation, throughput |
| `run_compare.py` | run all policies on shared demand, print the benchmark table |
| `train_drl.py` | train the PPO agent (resampled hotspot demand) and save the best checkpoint |
| `pipeline.py` | orchestrator: (inference) → run_compare |
| `stg{cn,at}_pred.npy` | model outputs, written here by `../ST*/run_infer.py` |
| `data/adj_mx_dijsk.pkl` | vendored road-network adjacency (copy of `../STGAT/data/METR-LA/`) |

## Data flow

```
stgcn_pred.npy ─┐
                ├─ ensemble(0.7/0.3, clamp) ─► node speed (mph)
stgat_pred.npy ─┘                                    │
adj_mx_dijsk.pkl ─ recover length d∝√(−ln adj) ──────┤
                                                     ▼
                       directed road graph  (length, t0, tpred, capacity)
                                                     │
                 OD demand — "hotspot" funnels many cars toward a few hub nodes
                          (this is what creates the herding pressure)
                                                     │
        ┌──────────────┬────────────────────┬──────────────────────┐
     static        prediction-greedy      load-aware          global-penalty
  (free-flow)        (HERDING)          (coordination)         (OURS, eq. 4)
        └──────────────┴────────────────────┴──────────────────────┘
                                                     ▼
        score every policy under the SAME realized BPR congestion model:
        ATT · Gini(edge load) · worst-link saturation ρ · throughput
```

The four policies are a deliberate **ablation**:

| policy | prediction? | coordination (load feedback)? | global penalty (eq. 4)? | role |
|---|:--:|:--:|:--:|---|
| `static` | ✗ | ✗ | ✗ | proposal baseline ① Dijkstra |
| `prediction-greedy` | ✓ | ✗ | ✗ | proposal baseline ②③ — **the herding case** |
| `load-aware` | ✓ | ✓ | ✗ | isolates the value of coordination |
| `global-penalty` | ✓ | ✓ | ✓ | the method (STGCN+STGAT + global penalty) |

## Result (default `hotspot` run)

```
policy                      |     ATT |    worst ρ | Gini(load)
static (free-flow Dijkstra) |  0.1118 |     4.5000 |    0.8985
prediction-greedy (HERDING) |  0.0349 |     2.3889 |    0.8526   <- everyone piles on the same links
load-aware (coord. only)    |  0.0226 |     0.3889 |    0.49xx
global-penalty (eq.4)       |  0.0229 |     0.2222 |    0.37xx   <- load spread out

Δ vs herding baseline:  Gini ↓ ~56% | worst-link ρ ↓ ~90% | ATT ↓ ~30%
```

Two things to read off this:

1. **It reproduces the proposal's predicted numbers** (section 五: "Gini ↓30%+,
   worst-link saturation ↓20%+, ATT ↓20–30% under burst load").
2. **The ablation shows what the global penalty buys.** Most of the ATT gain comes
   from *coordination* (`load-aware`). The **global penalty's own contribution is
   equity**: it pushes Gini and worst-link ρ down further for ~1% extra ATT — exactly
   the Wardrop *System-Optimum vs User-Equilibrium* trade-off the proposal cites
   (Wardrop 1952): give up a sliver of individual speed to kill the stampede.

(Exact numbers print from `run_compare.py`; the load-aware/global-penalty Gini values
shift slightly run-to-run with demand sampling but the ordering is stable.)

## Why there is no TSP / coordinate-embedding code here

An earlier prototype embedded the cost matrix into 2-D (MDS) and fed a pretrained
Euclidean-**TSP** solver. That approach was removed because:

- Navigation is **point-to-point routing on a graph**, not a closed salesman tour.
- A distance÷speed cost matrix is **non-metric**, so a 2-D embedding is lossy and the
  solver ends up optimizing distorted geometry while being scored on the true cost.
- The **global penalty / herding** — the project's novelty — needs many vehicles and a
  per-link saturation term, which a single TSP tour cannot express.

Everything here stays on the graph and routes many vehicles, so the herding effect is
something the model can actually exhibit and mitigate.

## The DRL agent (PPO) — scaffolded

`policies.py` now contains the interface for the proposal's learned decision module
(§4.4: POMDP + PPO + global-penalty reward). `policy_global_penalty` remains the
analytic **oracle** the agent should learn to match or beat.

- **`RoutingEnv`** — the POMDP. State = (graph, predicted edge times, current
  saturation, current node + destination); action = pick a neighbour; per-step reward
  = the negative eq. (4) generalized cost. Load persists across vehicles, so the
  penalty couples them (the herding-suppression signal). One episode = all vehicles.
- **`EGATActorCritic`** — trainable actor-critic (PyTorch), PPO-ready
  (`act` / `act_with_value` / `evaluate_actions`). The encoder is a minimal MLP
  placeholder — **swap it for the Residual E-GAT** (Lei et al. 2022), keeping the I/O
  `forward(feats[k,F]) -> (logits[k], value)`.
- **`PPOTrainer`** — clipped-surrogate PPO loop (eq. 5) with GAE. Add reward
  normalisation / orthogonal init / LR annealing for the proposal's "code-level
  optimisation".
- **`policy_drl(g, demand, agent)`** — rolls any agent out to paths, same contract as
  the other policies, so it drops into the comparison.

Use it in the benchmark:

```
python run_compare.py --drl placeholder    # analytic A*-greedy stand-in (runs now, no training)
python run_compare.py --drl path/to/agent.pt   # a trained EGATActorCritic checkpoint
```

Train the agent with `train_drl.py`. It resamples hotspot demand each episode,
evaluates on a fixed held-out demand against the herding/oracle references, and saves
the checkpoint with the best Gini-weighted score (0.25·ATT + 0.5·Gini + 0.25·worst-ρ):

```
python train_drl.py                                    # ~200 iters, CPU -> drl_agent.pt
python train_drl.py --iters 500 --train-vehicles 200   # longer run
python run_compare.py --drl drl_agent.pt               # benchmark the trained agent
```

`metrics.py` + `run_compare.py` are the **evaluation harness** — baselines and metrics
already wired. When SUMO is ready, swap the static **BPR** link model for SUMO's
microscopic feedback; the policy and metric code stay the same.

## Limitations

- Travel times are in **relative units** (the kernel's σ is unknown, so distances are
  relative). Compare ratios, not absolute seconds.
- **Capacity is uniform** (a proxy). The real Taichung network gets per-road capacity
  from OSM lane counts / road class.
- Only the **next-step** prediction is used (one snapshot), not the 15/30/60-min
  horizon — same simplification `run_infer.py` makes.
