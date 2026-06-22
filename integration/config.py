"""Single source of truth for the integration dataflow (prediction -> decision).

The STGCN/STGAT predictions are produced by the model repos' run_infer.py, which
write the .npy files into THIS folder; the decision stage reads them locally, so
the pipeline has no sideways dependency on any other folder.
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent          # .../integration
ROOT = HERE.parent                                # project root

# --- inputs (all local to this folder -> integration/ is fully standalone) ---
STGCN_PRED = HERE / "stgcn_pred.npy"             # written by ../STGCN/run_infer.py
STGAT_PRED = HERE / "stgat_pred.npy"             # written by ../STGAT/run_infer.py
# vendored copy of ../STGAT/data/METR-LA/adj_mx_dijsk.pkl (static road network;
# re-copy only if the network topology changes).
ADJ_PKL = HERE / "data" / "adj_mx_dijsk.pkl"

# --- speed / ensemble ---
SPEED_MIN, SPEED_MAX = 1.0, 70.0      # mph clamp
W_STGCN, W_STGAT = 0.7, 0.3           # ensemble weights
ADJ_THRESHOLD = 0.0                   # keep directed edges with kernel weight > this
KNN = 0                               # k-NN sparsification: keep each node's k nearest out-neighbors (0 = dense; k>0 reduces spreading headroom)

# --- volume-delay (BPR) congestion model: t(load) = t0 * (1 + A*(load/cap)^B) ---
#     The standard Bureau-of-Public-Roads link cost. It is what makes the
#     "herding effect" mechanical: piling vehicles on one link inflates its time.
BPR_A, BPR_B = 0.15, 4.0
EDGE_CAPACITY = 18.0                  # vehicles per link before saturation (uniform proxy)

# --- global-penalty reward weights (proposal eq. 4) ---
#     R = -alpha*travel_time - lambda1*sum max(0, rho - rho_th)^2 - lambda2*Var(rho)
ALPHA = 1.0
LAMBDA_SAT = 0.5                      # lambda1: per-link saturation-overflow penalty
LAMBDA_VAR = 0.8                      # lambda2: load-spread (variance) penalty (raised 0.3->0.8 to push Gini down)
RHO_THRESHOLD = 0.85                  # saturation threshold (proposal uses 0.85)
PENALTY_SCALE = 12.0                  # puts the penalty terms on the same scale as edge time

# --- experiment / demand ---
N_VEHICLES = 300
N_HOTSPOTS = 4                        # number of "city-center" sink nodes for the hotspot scenario
N_BATCHES = 30                        # incremental-assignment granularity (more = closer to system-optimal)
SCENARIO = "hotspot"                  # "random" | "hotspot" (hotspot funnels demand -> triggers herding)
SEED = 42
