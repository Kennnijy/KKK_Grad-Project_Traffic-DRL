"""The proposal's evaluation metrics (section 五 / 評估指標).

All policies are scored under the SAME realized BPR congestion model, given the
flows each policy produced — so the comparison is fair.

  - ATT                : average travel time over all vehicles (lower better)
  - TSTT               : total system travel time (lower better)
  - worst_rho          : worst-link saturation max(rho) (lower better)
  - frac_saturated     : fraction of used links above rho_threshold
  - gini_load          : Gini coefficient of edge load -> quantifies herding (lower = more even)
  - throughput_proxy   : served vehicles / ATT (higher better; a static-assignment proxy)
"""
import numpy as np

import config as C


def _bpr(t0, load, cap):
    return t0 * (1.0 + C.BPR_A * (load / cap) ** C.BPR_B)


def edge_loads(g, paths):
    """Vehicles per directed edge, plus how many vehicles were actually routed."""
    load = {e: 0.0 for e in g.edges()}
    served = 0
    for p in paths:
        if not p:
            continue
        served += 1
        for e in zip(p[:-1], p[1:]):
            load[e] += 1.0
    return load, served


def gini(values):
    """Gini coefficient of a non-negative vector (0 = perfectly even, ->1 = concentrated)."""
    x = np.sort(np.asarray(values, dtype=float))
    n = x.size
    if n == 0 or x.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float(np.sum((2 * idx - n - 1) * x) / (n * x.sum()))


def evaluate(g, paths, ref_edges):
    """Realized metrics for one policy. `ref_edges` is a fixed edge set shared
    across policies so the Gini comparison is apples-to-apples."""
    load, served = edge_loads(g, paths)

    tt_edge, rho_edge = {}, {}
    for e in g.edges():
        cap = g.edges[e]["cap"]
        tt_edge[e] = _bpr(g.edges[e]["t0"], load[e], cap)
        rho_edge[e] = load[e] / cap

    vehicle_times = []
    for p in paths:
        if not p:
            continue
        vehicle_times.append(sum(tt_edge[e] for e in zip(p[:-1], p[1:])))
    vt = np.array(vehicle_times) if vehicle_times else np.array([0.0])

    load_vec = np.array([load[e] for e in ref_edges]) if ref_edges else np.array([0.0])
    rho_vals = list(rho_edge.values()) or [0.0]

    att = float(vt.mean())
    return {
        "served": served,
        "att": att,
        "tstt": float(vt.sum()),
        "worst_rho": float(max(rho_vals)),
        "frac_saturated": float(np.mean([r > C.RHO_THRESHOLD for r in rho_vals])),
        "gini_load": gini(load_vec),
        "throughput_proxy": float(served / att) if att > 0 else 0.0,
    }
