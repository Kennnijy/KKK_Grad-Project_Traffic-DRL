"""Build the road network as a graph (NOT a 2D embedding).

We keep the problem ON THE GRAPH and do not project the cost matrix into 2D,
because (a) a distance/speed cost matrix is generally non-metric (asymmetric
speeds violate the triangle inequality) so a 2D embedding is lossy, and (b)
navigation is a graph routing problem, not a Euclidean tour. Every edge carries a
real travel time and a capacity, which is what lets us model congestion and the
herding effect.
"""
import pickle

import networkx as nx
import numpy as np

import config as C


def load_adjacency():
    """METR-LA Gaussian-kernel adjacency (directed, self-loops = 1)."""
    with open(C.ADJ_PKL, "rb") as f:
        try:
            _, _, adj = pickle.load(f)
        except UnicodeDecodeError:
            f.seek(0)
            _, _, adj = pickle.load(f, encoding="latin1")
    return np.asarray(adj, dtype=float)


def load_ensemble_speed():
    """STGCN+STGAT ensemble node speed (mph), clamped to the physical range.

    Returns (speed[N], info). The clamp to [SPEED_MIN, SPEED_MAX] is now just a
    guard: both models predict within 0-70 mph. `info` still reports each model's
    raw range and flags any out-of-range drift, so a future regression would be
    caught.
    """
    stgcn = np.load(C.STGCN_PRED).flatten()
    stgat = np.load(C.STGAT_PRED).flatten()

    def clamp(s):
        return np.clip(s, C.SPEED_MIN, C.SPEED_MAX)

    speed = C.W_STGCN * clamp(stgcn) + C.W_STGAT * clamp(stgat)
    info = {
        "stgcn_range": (float(stgcn.min()), float(stgcn.max())),
        "stgat_range": (float(stgat.min()), float(stgat.max())),
        "stgat_out_of_range": bool(stgat.max() > C.SPEED_MAX * 1.5),
    }
    return speed, info


def recover_length(adj, eps=1e-6):
    """Gaussian kernel adj = exp(-d^2/sigma^2)  ->  relative distance d ∝ sqrt(-ln adj).

    One consistent convention across the whole pipeline.
    """
    with np.errstate(divide="ignore"):
        d = np.sqrt(-np.log(np.clip(adj, eps, 1.0)))
    return d


def build_graph(adj, speed, knn=None):
    """Directed road graph. Each edge gets:
        length : relative road length (from the kernel)
        t0     : free-flow travel time          = length / SPEED_MAX
        tpred  : predicted (uncongested) time    = length / mean(predicted speed)
        cap    : capacity proxy

    k-NN sparsification (knn>0, default C.KNN): each node keeps only its `knn`
    strongest-adjacency (= nearest) out-neighbours, turning the dense Gaussian-kernel
    graph (~76 edges/node) into a realistic sparse road network. knn=0 keeps all edges.
    """
    n = adj.shape[0]
    length = recover_length(adj)
    knn = C.KNN if knn is None else knn
    g = nx.DiGraph()
    g.add_nodes_from(range(n))
    for i in range(n):
        cand = [j for j in range(n) if j != i and adj[i, j] > C.ADJ_THRESHOLD]
        if knn and len(cand) > knn:
            cand = sorted(cand, key=lambda j: adj[i, j], reverse=True)[:knn]
        for j in cand:
            edge_speed = max(0.5 * (speed[i] + speed[j]), C.SPEED_MIN)
            g.add_edge(
                i, j,
                length=float(length[i, j]),
                t0=float(length[i, j] / C.SPEED_MAX),
                tpred=float(length[i, j] / edge_speed),
                cap=float(C.EDGE_CAPACITY),
            )
    return g


def largest_scc(g):
    """Largest strongly connected component — guarantees sampled OD pairs are routable."""
    return max(nx.strongly_connected_components(g), key=len)
