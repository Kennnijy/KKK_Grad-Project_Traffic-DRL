# 基於機器學習與圖論的動態路網優化
**Dynamic Road-Network Optimization with STGCN / STGAT + Deep Reinforcement Learning**

A Tunghai University (東海大學) graduate project that combines spatio-temporal traffic
**prediction** (STGCN + STGAT) with a **deep-RL routing agent** to suppress the
*herding effect* (羊群效應) — the second-wave congestion that happens when a navigation
system funnels every vehicle that hears about a jam onto the same "fastest" detour.

The core idea, following the project proposal: predict near-future road speeds, turn
them into a congestion-weighted road graph, and route many vehicles with a **global
penalty** that trades a sliver of individual travel time for a far more even network
load — the Wardrop System-Optimum vs User-Equilibrium trade-off (Wardrop, 1952).

---

## Overview

The system has three modules (proposal §4):

1. **Prediction** — STGCN (spectral graph conv over the fixed road topology) + STGAT
   (graph attention for dynamic spatial dependencies) forecast per-sensor speed.
2. **Decision** — a PPO agent (Residual E-GAT actor-critic) routes vehicles
   neighbour-by-neighbour. Its reward is the proposal's **eq. (4) global penalty**:
   `R = −α·Δtravel_time − λ₁·Σ max(0, ρ−ρ_th)² − λ₂·Var(ρ)`, which punishes
   over-saturating any link and rewards spreading load evenly.
3. **Visualization** — SUMO + web dashboard (planned; see Roadmap).

Everything that connects prediction → decision lives in [`integration/`](integration/),
which is also a standalone evaluation harness for the routing policies.

---

## Architecture / data flow

```
 METR-LA  ──►  STGCN ─┐
              STGAT ─┴─► ensemble speed ─┐
                                         ├─►  congestion-weighted road graph
 adjacency (Gaussian kernel) ────────────┘     (per-edge travel time + capacity)
                                                       │
                          many vehicles (origin → destination demand)
                                                       │
        ┌───────────────┬──────────────────┬───────────────────────┐
     static          prediction-greedy   load-aware            global-penalty / DRL
   (Dijkstra)          (HERDING)         (coordination)        (eq. 4, anti-herding)
        └───────────────┴──────────────────┴───────────────────────┘
                                                       ▼
        evaluate under a BPR congestion model:
        ATT · Gini(edge load) · worst-link saturation ρ · throughput
```

---

## Repository structure

| Path | What it is |
|---|---|
| `STGAT/` | STGAT prediction model — cloned from [xyk0058/STGAT](https://github.com/xyk0058/STGAT) |
| `STGCN/` | STGCN prediction model — cloned from [hazdzz/STGCN](https://github.com/hazdzz/STGCN) |
| `DRL-and-graph-neural-network-for-routing-problems/` | Residual E-GAT / PPO routing base — cloned from [Lei-Kun/DRL-…-routing-problems](https://github.com/Lei-Kun/DRL-and-graph-neural-network-for-routing-problems) |
| **`integration/`** | **The core of this project**: prediction → decision pipeline, routing policies, the PPO/E-GAT agent, and the evaluation harness. See its [README](integration/README.md). |
| `*.pdf` | The three method papers (STGCN, STGAT, Residual E-GAT) — see Credits |
| `requirements.txt` | Python dependencies (pip freeze of the `traffic_rl` env) |

> Note: `STGAT/`, `STGCN/`, and `DRL-...` are upstream repositories with their own git
> history; our work is concentrated in `integration/`.

---

## Setup

Python 3.10, CUDA 12.1. A GPU is needed to (re)train the models and the DRL agent; the
routing comparison itself runs on CPU in ~1 s.

```bash
conda create -n traffic_rl python=3.10
conda activate traffic_rl
pip install -r requirements.txt        # torch 2.3.1+cu121, torch_geometric, networkx, scipy, ...
```

---

## Usage

### 1. Traffic prediction → `integration/`
Each model writes its predictions into `integration/`:
```bash
cd STGCN && python run_infer.py     # -> ../integration/stgcn_pred.npy
cd STGAT && python run_infer.py     # -> ../integration/stgat_pred.npy
# (re-train if needed: cd STGAT && python train.py --cuda --epoch 500 --early_stop_maxtry 40)
```

### 2. Routing comparison (the herding experiment)
```bash
cd integration
python pipeline.py --skip-infer            # reuse cached predictions, run the full comparison
python run_compare.py --scenario hotspot   # or --scenario random / --vehicles N / --capacity C
```

### 3. Train & benchmark the DRL agent
```bash
cd integration
python train_drl.py                        # PPO + Residual E-GAT; auto-uses CUDA if available
python run_compare.py --drl drl_agent.pt   # add the trained agent as a 5th policy
```
See [`integration/README.md`](integration/README.md) for the policy/metric details and tuning knobs.

---

## Results (latest `integration/results.json`)

Hotspot scenario, 300 vehicles funnelled toward 4 hub nodes (lower is better; travel
times are in relative units). The **prediction-greedy** row is the herding baseline.

| Policy | ATT | worst-link ρ | Gini(load) |
|---|---:|---:|---:|
| Static Dijkstra (no prediction) | 0.112 | 4.50 | 0.920 |
| Prediction-greedy (**herding baseline**) | 0.035 | 2.39 | 0.884 |
| Load-aware (coordination only) | 0.023 | 0.39 | 0.598 |
| **Global-penalty (eq. 4 oracle)** | 0.024 | **0.22** | **0.436** |
| DRL agent (PPO + E-GAT) | 0.023 | 0.17 | 0.574 |

Versus the herding baseline, the **global-penalty** method cuts **Gini ≈ 51%**,
**worst-link saturation ≈ 91%**, and **ATT ≈ 33%** — meeting/exceeding the proposal's
targets (Gini ↓30%+, worst-link ↓20%+, ATT ↓20–30%). The learned **DRL agent** already
matches the oracle on ATT and worst-link ρ; closing the remaining **Gini** gap to the
oracle is the current focus.

---

## Status & roadmap

- [x] STGCN / STGAT reproduced on METR-LA (STGAT test MAE ≈ 3.16, beats the proposal target)
- [x] Prediction → decision pipeline + analytic baselines (`integration/`)
- [x] Global-penalty oracle demonstrating herding mitigation (eq. 4)
- [x] PPO + Residual E-GAT routing agent (GPU-enabled)
- [ ] Close the DRL agent's Gini gap to the oracle (reward/representation tuning)
- [ ] SUMO microscopic simulation + TraCI integration
- [ ] Web dashboard (event injection, live re-routing visualization)
- [ ] Field scenario: Tunghai University → Taichung Station road network (OSM)

---

## Team

東海大學 畢業專題 — **S12350312 黃子修 · S12350302 黃少鯤 · S12350131 江彥萱**

## Credits & references

This project builds on three open-source implementations; the corresponding papers are
included in this folder as PDFs.

| Module | Code (cloned from) | Paper |
|---|---|---|
| STGCN prediction | [hazdzz/STGCN](https://github.com/hazdzz/STGCN) | Yu, Yin & Zhu — *Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting*, IJCAI 2018 |
| STGAT prediction | [xyk0058/STGAT](https://github.com/xyk0058/STGAT) | Kong et al. — *STGAT: Spatial-Temporal Graph Attention Networks for Traffic Flow Forecasting*, IEEE Access 2020 |
| DRL routing | [Lei-Kun/DRL-and-graph-neural-network-for-routing-problems](https://github.com/Lei-Kun/DRL-and-graph-neural-network-for-routing-problems) | Lei et al. — *Solve routing problems with a residual edge-graph attention neural network*, Neurocomputing 2022 |

Additional methods referenced: Schulman et al. 2017 (PPO) · Wardrop 1952
(User-Equilibrium vs System-Optimum) · Lopez et al. 2018 (SUMO).
#
