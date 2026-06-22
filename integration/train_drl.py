#!/usr/bin/env python3
"""Train the PPO routing agent (proposal §4.4) and save a checkpoint.

Pipeline:
    build graph (from the vendored adjacency + STGCN/STGAT predictions)
      -> RoutingEnv with eq.(4) reward + arrival/fail shaping + per-episode demand
      -> EGATActorCritic trained by PPOTrainer (clipped surrogate, eq.5)
      -> periodic evaluation on a FIXED held-out hotspot demand
      -> save the checkpoint with the best Gini-weighted score

Design choices (confirmed):
  * demand is RESAMPLED every episode (hotspot scenario) so the agent generalizes
    over OD pairs rather than memorizing one instance.
  * reward shaping is ON: an arrival bonus and a stuck/max-hops penalty (both scaled
    by the mean free-flow edge time) are added on top of the eq.(4) per-step cost.
    The evaluation metrics (ATT/Gini/worst-rho via metrics.py) are computed
    independently, so they remain faithful to the proposal.
  * "best" checkpoint = highest Gini-weighted relative improvement vs the
    prediction-greedy HERDING baseline (0.25*ATT + 0.5*Gini + 0.25*worst-rho),
    discounted if the agent fails to route enough vehicles.

The trained checkpoint plugs straight into the benchmark:
    python run_compare.py --drl drl_agent.pt

Usage:
    python train_drl.py                       # sensible defaults (~200 iters, CPU)
    python train_drl.py --iters 500 --train-vehicles 200 --out drl_agent.pt
"""
import argparse

import numpy as np

import config as C
import metrics as M
import network as net
import policies as pol

try:
    import torch
except ImportError:                       # pragma: no cover
    raise SystemExit("train_drl.py requires PyTorch.  pip install torch")


def hotspot_demand(scc, hubs, n, rng):
    """n vehicles: random origins in the SCC, destinations funneled to the hubs."""
    origins = rng.choice(scc, size=n)
    dests = rng.choice(hubs, size=n)
    return [(int(o), int(d)) for o, d in zip(origins, dests) if o != d]


def build_eval(g, eval_demand):
    """Fixed reference for scoring every checkpoint the same way:
    the analytic policies define a stable edge set (for Gini) and the
    prediction-greedy herding metrics we measure improvement against."""
    base = {
        "static": pol.policy_static(g, eval_demand),
        "herding": pol.policy_prediction_greedy(g, eval_demand),
        "load_aware": pol.policy_load_aware(g, eval_demand),
        "global_penalty": pol.policy_global_penalty(g, eval_demand),
    }
    ref = set()
    for paths in base.values():
        load, _ = M.edge_loads(g, paths)
        ref |= {e for e, v in load.items() if v > 0}
    ref = sorted(ref)
    return ref, M.evaluate(g, base["herding"], ref), M.evaluate(g, base["global_penalty"], ref)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iters", type=int, default=200, help="PPO iterations (episodes)")
    ap.add_argument("--train-vehicles", type=int, default=300, help="vehicles per training episode")
    ap.add_argument("--eval-vehicles", type=int, default=C.N_VEHICLES, help="vehicles in the held-out eval")
    ap.add_argument("--eval-every", type=int, default=25, help="evaluate + maybe checkpoint every N iters")
    ap.add_argument("--log-every", type=int, default=10, help="print a training line every N iters")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--entropy-coef", type=float, default=0.03,
                    help="PPO entropy bonus (higher = more exploration; raised 0.01->0.03)")
    ap.add_argument("--max-hops", type=int, default=60, help="give up a trip after this many hops")
    ap.add_argument("--arrival-mult", type=float, default=2.0, help="arrival bonus = mult * mean free-flow edge time")
    ap.add_argument("--fail-mult", type=float, default=5.0, help="fail penalty = mult * mean free-flow edge time")
    ap.add_argument("--min-served", type=float, default=0.95, help="served-fraction target for full score credit")
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"),
                    help="cuda or cpu (default: cuda if available)")
    ap.add_argument("--out", default=str(C.HERE / "drl_agent.pt"), help="checkpoint path for the best agent")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    if str(args.device).startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)

    # --- network (same inputs as run_compare) ---
    adj = net.load_adjacency()
    speed, info = net.load_ensemble_speed()
    g = net.build_graph(adj, speed)
    scc = sorted(net.largest_scc(g))
    hubs = sorted(scc, key=lambda n: g.in_degree(n), reverse=True)[:C.N_HOTSPOTS]
    t_ref = float(np.mean([g.edges[e]["t0"] for e in g.edges()]))

    print(f"Graph {g.number_of_nodes()} nodes / {g.number_of_edges()} edges; "
          f"SCC {len(scc)}; hubs {hubs}; mean free-flow edge time t_ref={t_ref:.4f}")
    print(f"Ensemble speed {speed.min():.0f}-{speed.max():.0f} mph "
          f"(STGAT {info['stgat_range'][0]:.0f}-{info['stgat_range'][1]:.0f})")

    # --- fixed held-out evaluation demand + reference baselines ---
    eval_rng = np.random.default_rng(args.seed + 9973)
    eval_demand = hotspot_demand(scc, hubs, args.eval_vehicles, eval_rng)
    eval_ref, herding_m, oracle_m = build_eval(g, eval_demand)
    base_att, base_gini, base_rho = herding_m["att"], herding_m["gini_load"], herding_m["worst_rho"]
    print(f"\nEval demand: {len(eval_demand)} vehicles (fixed). Reference on this demand:")
    print(f"  herding  baseline : ATT {base_att:.4f}  Gini {base_gini:.3f}  worst-rho {base_rho:.3f}")
    print(f"  oracle (eq.4)     : ATT {oracle_m['att']:.4f}  Gini {oracle_m['gini_load']:.3f}  "
          f"worst-rho {oracle_m['worst_rho']:.3f}  <- the agent's target to match/beat\n")

    # --- env (resampled hotspot demand each episode) + agent + PPO ---
    train_rng = np.random.default_rng(args.seed)
    env = pol.RoutingEnv(
        g,
        demand_fn=lambda: hotspot_demand(scc, hubs, args.train_vehicles, train_rng),
        use_penalty=True,
        max_hops=args.max_hops,
        arrival_bonus=args.arrival_mult * t_ref,
        fail_penalty=args.fail_mult * t_ref,
    )
    agent = pol.EGATActorCritic(g).to(args.device)
    trainer = pol.PPOTrainer(env, agent, lr=args.lr, entropy_coef=args.entropy_coef)

    def evaluate_agent():
        agent.eval()
        paths = pol.policy_drl(g, eval_demand, agent)
        m = M.evaluate(g, paths, eval_ref)
        agent.train()
        served = m["served"] / max(1, len(eval_demand))

        def rel(b, x):
            return (b - x) / b if b else 0.0
        # weighted toward Gini (herding suppression), per the scoring choice
        improv = float(0.25 * rel(base_att, m["att"])
                       + 0.50 * rel(base_gini, m["gini_load"])
                       + 0.25 * rel(base_rho, m["worst_rho"]))
        score = improv - 2.0 * max(0.0, args.min_served - served)
        return score, m, served

    # --- training loop ---
    best = -float("inf")
    agent.train()
    print(f"Training {args.iters} iters on {args.device} (train-vehicles={args.train_vehicles}); "
          f"saving best -> {args.out}\n")
    for it in range(1, args.iters + 1):
        traj = trainer.collect_episode()
        stats = trainer.update(traj)
        if it % args.eval_every == 0 or it == args.iters:
            score, m, served = evaluate_agent()
            tag = ""
            if score > best:
                best = score
                agent.save(args.out)
                tag = "  <- saved best"
            print(f"iter {it:4d} | ep_return {sum(t['reward'] for t in traj):8.2f} "
                  f"| ATT {m['att']:.4f}  Gini {m['gini_load']:.3f}  worst-rho {m['worst_rho']:.3f}  "
                  f"served {served * 100:3.0f}% | score {score:+.3f}{tag}")
        elif it % args.log_every == 0:
            print(f"iter {it:4d} | ep_return {sum(t['reward'] for t in traj):8.2f} "
                  f"| pi_loss {stats.get('policy_loss', 0):+.4f}  v_loss {stats.get('value_loss', 0):.4f}")

    print(f"\nDone. Best score {best:+.3f}. Best agent -> {args.out}")
    print(f"Benchmark it with:  python run_compare.py --drl {args.out}")


if __name__ == "__main__":
    main()
