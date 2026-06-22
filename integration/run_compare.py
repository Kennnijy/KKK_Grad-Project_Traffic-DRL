#!/usr/bin/env python3
"""Compare routing strategies on the SAME predicted network and demand.

Reproduces the proposal's M4 benchmark idea (section 五): Dijkstra vs
prediction+static vs our method, scored by ATT, Gini of edge load, and
worst-link saturation — the metrics the proposal uses to quantify the
herding effect (羊群效應) and its mitigation.

    cd integration && python run_compare.py
    python run_compare.py --scenario random --vehicles 200
"""
import argparse
import json

import numpy as np

import config as C
import metrics as M
import network as net
import policies as pol


def make_demand(g, scc, rng):
    """Origin-destination pairs.

    'hotspot' funnels many origins into a few high-in-degree hub nodes (think
    rush-hour into the city centre / an arterial closure pushing everyone toward
    the same detour). That concentration is what triggers the herding effect.
    """
    scc = sorted(scc)
    hubs = sorted(scc, key=lambda n: g.in_degree(n), reverse=True)[:C.N_HOTSPOTS]
    origins = rng.choice(scc, size=C.N_VEHICLES)
    if C.SCENARIO == "random":
        dests = rng.choice(scc, size=C.N_VEHICLES)
    else:  # hotspot
        dests = rng.choice(hubs, size=C.N_VEHICLES)
    demand = [(int(o), int(d)) for o, d in zip(origins, dests) if o != d]
    return demand, hubs


def fmt(results, baseline_key):
    """Comparison table with % change vs the herding baseline."""
    cols = [
        ("ATT", "att", "low"),
        ("TSTT", "tstt", "low"),
        ("worst ρ", "worst_rho", "low"),
        ("saturated%", "frac_saturated", "low"),
        ("Gini(load)", "gini_load", "low"),
        ("throughput*", "throughput_proxy", "high"),
    ]
    name_w = max(len(n) for n in results)
    head = f"{'policy':<{name_w}} | " + " | ".join(f"{c[0]:>11}" for c in cols)
    lines = [head, "-" * len(head)]
    base = results[baseline_key]
    for name, r in results.items():
        cells = []
        for label, key, _ in cols:
            v = r[key]
            s = f"{v:11.4f}" if abs(v) < 1000 else f"{v:11.1f}"
            cells.append(s)
        lines.append(f"{name:<{name_w}} | " + " | ".join(cells))
    # deltas vs baseline for the headline metrics
    lines.append("")
    lines.append(f"Δ vs '{baseline_key}' (negative = improvement):")
    for name, r in results.items():
        if name == baseline_key:
            continue
        d_att = 100 * (r["att"] - base["att"]) / base["att"] if base["att"] else 0
        d_gini = 100 * (r["gini_load"] - base["gini_load"]) / base["gini_load"] if base["gini_load"] else 0
        d_rho = 100 * (r["worst_rho"] - base["worst_rho"]) / base["worst_rho"] if base["worst_rho"] else 0
        lines.append(f"  {name:<{name_w}} : ATT {d_att:+6.1f}% | "
                     f"Gini {d_gini:+6.1f}% | worst ρ {d_rho:+6.1f}%")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", choices=["random", "hotspot"], default=C.SCENARIO)
    ap.add_argument("--vehicles", type=int, default=C.N_VEHICLES)
    ap.add_argument("--capacity", type=float, default=C.EDGE_CAPACITY)
    ap.add_argument("--drl", default=None, metavar="placeholder|CKPT.pt",
                    help="add the DRL agent as a 5th policy: 'placeholder' (analytic "
                         "stand-in, no training needed) or a path to a trained checkpoint")
    args = ap.parse_args()
    C.SCENARIO, C.N_VEHICLES, C.EDGE_CAPACITY = args.scenario, args.vehicles, args.capacity

    rng = np.random.default_rng(C.SEED)
    adj = net.load_adjacency()
    speed, info = net.load_ensemble_speed()
    g = net.build_graph(adj, speed)
    scc = net.largest_scc(g)

    print(f"Graph: {g.number_of_nodes()} nodes, {g.number_of_edges()} directed edges; "
          f"largest SCC = {len(scc)} nodes")
    print(f"Ensemble speed {speed.min():.1f}~{speed.max():.1f} mph "
          f"(STGCN {info['stgcn_range'][0]:.0f}-{info['stgcn_range'][1]:.0f}, "
          f"STGAT {info['stgat_range'][0]:.0f}-{info['stgat_range'][1]:.0f})")
    if info["stgat_out_of_range"]:
        print("  ⚠ STGAT predictions exceed the physical speed range and were clamped "
              f"to {C.SPEED_MAX} mph (prediction-module issue — see README).")

    demand, hubs = make_demand(g, scc, rng)
    print(f"Scenario '{C.SCENARIO}': {len(demand)} vehicles, capacity={C.EDGE_CAPACITY}/edge, "
          f"hubs={hubs}\n")

    runs = {
        "static (free-flow Dijkstra)": pol.policy_static(g, demand),
        "prediction-greedy (HERDING)": pol.policy_prediction_greedy(g, demand),
        "load-aware (coord. only)": pol.policy_load_aware(g, demand),
        "global-penalty (OURS, eq.4)": pol.policy_global_penalty(g, demand),
    }

    # DRL agent slot (proposal §4.4). Off by default; opt in with --drl.
    if args.drl:
        agent = pol.make_drl_agent(args.drl, g)
        label = ("drl-agent (placeholder)" if args.drl in ("placeholder", "oracle")
                 else f"drl-agent ({args.drl})")
        runs[label] = pol.policy_drl(g, demand, agent)
    else:
        print("(DRL slot scaffolded but inactive — add it with "
              "`--drl placeholder` or `--drl path/to/checkpoint.pt`.)\n")

    # fixed reference edge set = union of links used by any policy (fair Gini)
    ref = set()
    for paths in runs.values():
        load, _ = M.edge_loads(g, paths)
        ref |= {e for e, v in load.items() if v > 0}
    ref = sorted(ref)

    results = {name: M.evaluate(g, paths, ref) for name, paths in runs.items()}
    print(fmt(results, baseline_key="prediction-greedy (HERDING)"))
    print("\n  * throughput proxy = served vehicles / ATT (static-assignment proxy, not SUMO).")

    out = {
        "scenario": C.SCENARIO,
        "n_vehicles": len(demand),
        "capacity": C.EDGE_CAPACITY,
        "hubs": hubs,
        "stgat_out_of_range": info["stgat_out_of_range"],
        "results": results,
    }
    with open(C.HERE / "results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {C.HERE / 'results.json'}")


if __name__ == "__main__":
    main()
