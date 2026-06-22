"""Routing policies, framed as point-to-point navigation for many vehicles.

This is the heart of the strategy difference. The proposal's problem is:
many vehicles route from origin to destination, and naive "everyone takes the
predicted-fastest road" behaviour creates the herding effect (羊群效應). We model
exactly that, as an ablation:

  1. static            - free-flow shortest path, prediction ignored        (baseline 1)
  2. prediction_greedy - predicted-fastest path, per vehicle, NO coordination (baseline 2/3, HERDING)
  3. load_aware        - incremental load-aware assignment, no global penalty (ablation: coordination only)
  4. global_penalty    - load-aware assignment WITH proposal eq. (4)         (our method)

A policy returns one node-path per vehicle (or None if unreachable).
"""
import abc
from collections import namedtuple

import networkx as nx
import numpy as np

import config as C


def _route_fixed(g, demand, weight):
    """Shortest path for every (o, d) under a FIXED edge weight.

    No load feedback: every vehicle sees the same costs, so they pile onto the
    same links. This is precisely the mechanism behind the herding effect.
    """
    paths = []
    for o, d in demand:
        try:
            paths.append(nx.shortest_path(g, o, d, weight=weight))
        except nx.NetworkXNoPath:
            paths.append(None)
    return paths


def policy_static(g, demand):
    """Baseline 1: free-flow shortest path (Dijkstra), prediction ignored."""
    return _route_fixed(g, demand, "t0")


def policy_prediction_greedy(g, demand):
    """Baseline 2/3: STGCN+STGAT predicted-fastest path, per vehicle, uncoordinated."""
    return _route_fixed(g, demand, "tpred")


def _bpr(t0, load, cap):
    """Bureau-of-Public-Roads volume-delay function."""
    return t0 * (1.0 + C.BPR_A * (load / cap) ** C.BPR_B)


def policy_incremental(g, demand, use_penalty, n_batches=None):
    """Incremental load-aware assignment.

    Vehicles are assigned in batches; before each batch the edge cost is
    recomputed from the load left by earlier batches, so later vehicles are
    steered onto alternatives. This is the transparent stand-in for a PPO agent
    trained with the global-penalty reward (see README) and the evaluation
    harness that agent will later be scored against.

    Generalized edge cost realizes proposal eq. (4):
        cost_e = alpha * t_e(load)
               + lambda1 * t_ref * scale * max(0, rho_e - rho_th)^2   # saturation overflow
               + lambda2 * t_ref * scale * max(0, rho_e - mean_rho)   # per-edge surrogate of Var(rho)

    With use_penalty=False the lambda terms drop out, leaving plain congestion-aware
    assignment (the ablation that isolates the global penalty's marginal effect).
    """
    n_batches = n_batches or C.N_BATCHES
    edges = list(g.edges())
    t0 = {e: g.edges[e]["t0"] for e in edges}
    cap = {e: g.edges[e]["cap"] for e in edges}
    load = {e: 0.0 for e in edges}
    t_ref = float(np.mean([t0[e] for e in edges])) if edges else 1.0

    paths = [None] * len(demand)
    for batch in np.array_split(np.arange(len(demand)), n_batches):
        rho = {e: load[e] / cap[e] for e in edges}
        mean_rho = float(np.mean([rho[e] for e in edges])) if edges else 0.0
        for e in edges:
            cost = C.ALPHA * _bpr(t0[e], load[e], cap[e])
            if use_penalty:
                overflow = max(0.0, rho[e] - C.RHO_THRESHOLD) ** 2
                spread = max(0.0, rho[e] - mean_rho)
                cost += t_ref * C.PENALTY_SCALE * (
                    C.LAMBDA_SAT * overflow + C.LAMBDA_VAR * spread
                )
            g.edges[e]["cost"] = cost

        for k in batch:
            o, d = demand[k]
            try:
                p = nx.shortest_path(g, o, d, weight="cost")
            except nx.NetworkXNoPath:
                p = None
            paths[k] = p
            if p:
                for e in zip(p[:-1], p[1:]):
                    load[e] += 1.0
    return paths


def policy_load_aware(g, demand):
    """Ablation: coordination via congestion feedback only (no global penalty)."""
    return policy_incremental(g, demand, use_penalty=False)


def policy_global_penalty(g, demand):
    """Our method: congestion feedback + proposal eq. (4) global penalty."""
    return policy_incremental(g, demand, use_penalty=True)


# ===================================================================
# DRL / PPO scaffold   (proposal §4.4: POMDP + PPO + global-penalty reward)
# -------------------------------------------------------------------
# The policies above are analytic. The proposal's decision module is a *learned*
# PPO agent (Residual E-GAT actor + pointer decoder) whose action is "pick the next
# neighbour node" and whose reward is eq. (4). This section scaffolds that interface
# so a trained agent slots straight into run_compare with no other changes:
#
#   RoutingEnv        - the POMDP the agent acts in / trains against (reward = eq.4)
#   DRLRoutingAgent   - the agent contract the rollout depends on (.act)
#   GreedyOracleAgent - analytic placeholder so the whole slot runs *today*
#   EGATActorCritic   - trainable Residual E-GAT actor-critic (PyTorch; PPO-ready)
#   PPOTrainer        - PPO loop skeleton (clipped surrogate, eq.5)
#   policy_drl        - rolls an agent out to produce paths, like the other policies
#
# `policy_global_penalty` is the analytic ORACLE the agent should learn to match or
# beat; this env's per-step reward is exactly that eq.(4) generalized cost.
# ===================================================================

# Per-candidate edge features for the learned policy:
#   [t0, tpred, rho, is_dest, dist_to_dest, mean_rho, rho-mean_rho, overflow]
# The last three expose eq. (4)'s load-spread / saturation signal, so the agent can
# learn to even out load (suppress herding / lower Gini), not just shorten trips.
EDGE_FEATURE_DIM = 8
_BIG = 1.0e3  # finite stand-in for "unreachable" distance in network features

# One decision point: at `node`, heading to `dest`, choose one of `neighbors`.
#   feats : [k, EDGE_FEATURE_DIM] per-candidate features for a learned policy
#   gcost : [k] eq.(4) generalized cost of each candidate edge (for the oracle)
#   to_go : [k] shortest free-flow time from each candidate to dest (A*-style hint)
Observation = namedtuple(
    "Observation",
    ["node", "dest", "neighbors", "feats", "gcost", "to_go",
     "node_dyn", "edge_rho", "vehicle"])   # last 3: per-vehicle E-GAT encoder inputs


class RoutingEnv:
    """Sequential multi-vehicle routing POMDP (proposal §4.4).

    State  s_t = (G, X_pred, rho_t, o_t): graph + predicted edge times + current
                 saturation + (current node, destination).
    Action a_t in N(v_t): pick a neighbour of the current node.
    Reward r_t = -(eq.4 generalized cost of the chosen edge):
                 -(alpha*t_e(load) + lambda1*t_ref*scale*max(0,rho-rho_th)^2
                                   + lambda2*t_ref*scale*max(0,rho-mean_rho)).

    Vehicles are routed one after another and the load PERSISTS across them, so the
    penalty couples vehicles — that coupling is what suppresses the herding effect.
    One episode = routing every vehicle in `demand`.

    Optional training-time shaping (default off -> reward stays faithful to eq.4):
    `arrival_bonus` is added when a vehicle reaches its destination and
    `fail_penalty` subtracted if it gets stuck. Pass `demand_fn` (a zero-arg callable
    returning a demand list) to resample demand on every reset().
    """

    def __init__(self, g, demand=None, use_penalty=True, max_hops=60,
                 arrival_bonus=0.0, fail_penalty=0.0, demand_fn=None):
        self.g = g
        self.demand_fn = demand_fn
        self.demand = list(demand) if demand is not None else None
        self.use_penalty = use_penalty
        self.max_hops = max_hops
        self.arrival_bonus = arrival_bonus
        self.fail_penalty = fail_penalty
        self.succ = {v: list(g.successors(v)) for v in g.nodes()}
        self._t0 = {e: g.edges[e]["t0"] for e in g.edges()}
        self._tpred = {e: g.edges[e]["tpred"] for e in g.edges()}
        self.cap = float(C.EDGE_CAPACITY)
        self.n_edges = max(1, g.number_of_edges())
        self.N = g.number_of_nodes()
        self.edge_list = list(g.edges())          # canonical edge order (matches the agent's edge_index)
        self.t_ref = float(np.mean(list(self._t0.values()))) if self._t0 else 1.0
        self._rg = g.reverse(copy=False)          # for shortest free-flow time to dest
        self._dist_cache = {}
        self._enc = None                          # per-vehicle (node_dyn, edge_rho) for the E-GAT encoder
        self.reset()

    # --- load / saturation bookkeeping ---
    def _rho(self, e):
        return self.load.get(e, 0.0) / self.cap

    @property
    def _mean_rho(self):
        return self.total_load / (self.n_edges * self.cap)

    def _gcost(self, u, v):
        """eq.(4) marginal cost of traversing (u, v) at the current load."""
        e = (u, v)
        cost = C.ALPHA * _bpr(self._t0[e], self.load.get(e, 0.0), self.cap)
        if self.use_penalty:
            rho = self._rho(e)
            overflow = max(0.0, rho - C.RHO_THRESHOLD) ** 2
            spread = max(0.0, rho - self._mean_rho)
            cost += self.t_ref * C.PENALTY_SCALE * (
                C.LAMBDA_SAT * overflow + C.LAMBDA_VAR * spread)
        return cost

    def _dist_to_dest(self, dest):
        d = self._dist_cache.get(dest)
        if d is None:
            d = nx.single_source_dijkstra_path_length(self._rg, dest, weight="t0")
            self._dist_cache[dest] = d
        return d

    def _compute_enc_ctx(self):
        """Per-vehicle E-GAT encoder inputs: node features [N,3] and edge rho [E]
        (snapshotted at the start of each vehicle's trip)."""
        dist = self._dist_to_dest(self._dest)
        finite = [d for d in dist.values() if np.isfinite(d)]
        dmax = max(finite) if finite else 1.0
        node_dyn = np.zeros((self.N, 3), dtype=np.float32)
        for v in range(self.N):
            d = dist.get(v, dmax * 2.0)
            node_dyn[v, 0] = min(d, dmax * 2.0) / (dmax + 1e-9)    # dist-to-dest (normalized)
            node_dyn[v, 1] = 1.0 if v == self._dest else 0.0        # is-dest
            outs = self.succ[v]
            if outs:
                s = sum(self.load.get((v, w), 0.0) for w in outs)
                node_dyn[v, 2] = s / (len(outs) * self.cap)          # mean out-rho
        edge_rho = np.empty(len(self.edge_list), dtype=np.float32)
        for i, e in enumerate(self.edge_list):
            edge_rho[i] = self.load.get(e, 0.0) / self.cap
        return node_dyn, edge_rho

    # --- episode control ---
    def reset(self):
        if self.demand_fn is not None:
            self.demand = list(self.demand_fn())
        if self.demand is None:
            raise ValueError("RoutingEnv requires `demand` or `demand_fn`.")
        self.load = {}
        self.total_load = 0.0
        self.paths = [None] * len(self.demand)
        self._vi = -1                 # current vehicle index
        self._cur = self._dest = None
        self._visited = None
        self._hops = 0
        self.done = False
        self._start_next_vehicle()
        return self._observe()

    def _valid_neighbors(self):
        return [w for w in self.succ[self._cur] if w not in self._visited]

    def _start_next_vehicle(self):
        """Advance to the next vehicle that actually has a choice to make."""
        while True:
            self._vi += 1
            if self._vi >= len(self.demand):
                self.done = True
                self._cur = None
                return
            o, d = self.demand[self._vi]
            self._cur, self._dest = o, d
            self._visited = {o}
            self._hops = 0
            if o == d or not self._valid_neighbors():
                self.paths[self._vi] = None       # trivial or dead-end -> failed trip
                continue
            self.paths[self._vi] = [o]
            self._enc = self._compute_enc_ctx()   # snapshot encoder inputs for this vehicle
            return

    def _observe(self):
        if self.done:
            return None
        nbrs = self._valid_neighbors()
        dist = self._dist_to_dest(self._dest)
        mean_rho = self._mean_rho
        feats = np.zeros((len(nbrs), EDGE_FEATURE_DIM), dtype=np.float32)
        gcost = np.zeros(len(nbrs), dtype=np.float32)
        to_go = np.full(len(nbrs), np.inf, dtype=np.float32)
        for i, w in enumerate(nbrs):
            e = (self._cur, w)
            tg = dist.get(w, np.inf)
            to_go[i] = tg
            gcost[i] = self._gcost(self._cur, w)
            rho = self._rho(e)
            feats[i] = (self._t0[e], self._tpred[e], rho,
                        1.0 if w == self._dest else 0.0, min(tg, _BIG),
                        mean_rho, rho - mean_rho, max(0.0, rho - C.RHO_THRESHOLD))
        node_dyn, edge_rho = self._enc
        return Observation(self._cur, self._dest, nbrs, feats, gcost, to_go,
                           node_dyn, edge_rho, self._vi)

    def step(self, action_index):
        """Apply chosen neighbour (index into obs.neighbors).
        Returns (next_obs, reward, done, info)."""
        nbrs = self._valid_neighbors()
        u, w = self._cur, nbrs[action_index]
        reward = -self._gcost(u, w)
        e = (u, w)
        self.load[e] = self.load.get(e, 0.0) + 1.0
        self.total_load += 1.0
        self._visited.add(w)
        self._cur = w
        self._hops += 1
        self.paths[self._vi].append(w)
        reached = (w == self._dest)
        stuck = (self._hops >= self.max_hops) or (not self._valid_neighbors())
        if reached:
            reward += self.arrival_bonus
        elif stuck:
            reward -= self.fail_penalty
        info = {"vehicle": self._vi, "reached_dest": reached}
        if reached or stuck:
            if not reached:
                self.paths[self._vi] = None       # failed before reaching dest
            self._start_next_vehicle()
        return self._observe(), reward, self.done, info


class DRLRoutingAgent(abc.ABC):
    """Contract that the rollout (policy_drl) needs from any routing agent."""

    @abc.abstractmethod
    def act(self, obs, greedy=False):
        """Return an action index into obs.neighbors."""

    def load(self, path):     # analytic agents need no checkpoint
        raise NotImplementedError

    def save(self, path):
        raise NotImplementedError


class GreedyOracleAgent(DRLRoutingAgent):
    """Analytic placeholder: A*-greedy on eq.(4) cost + free-flow time-to-go.

    A myopic (1-step) stand-in for the trained policy so policy_drl and the whole
    comparison run *today*. It is NOT learned; replace it with a trained
    EGATActorCritic via run_compare's --drl flag. Being myopic, it should
    under-perform the batched `policy_global_penalty` oracle — which is the point:
    it proves the slot works and gives the PPO agent a concrete bar to clear.
    """

    def act(self, obs, greedy=True):
        return int(np.argmin(obs.gcost + obs.to_go))


try:
    import torch
    import torch.nn as nn
    from torch.distributions import Categorical
    from torch_geometric.nn import GATv2Conv
    _TORCH_OK = True
except Exception:                     # torch / torch_geometric optional for analytic policies
    _TORCH_OK = False


if _TORCH_OK:

    class _EGATEncoder(nn.Module):
        """Residual edge-aware GAT encoder (Lei et al. 2022 style): message passing
        over the road graph with edge features, residual connections + LayerNorm."""

        def __init__(self, node_dim, edge_dim, hidden, layers=3, heads=4):
            super().__init__()
            self.in_proj = nn.Linear(node_dim, hidden)
            self.in_norm = nn.LayerNorm(hidden)
            self.convs = nn.ModuleList([
                GATv2Conv(hidden, hidden, heads=heads, concat=False,
                          edge_dim=edge_dim, add_self_loops=False)
                for _ in range(layers)])
            self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])

        def forward(self, x, edge_index, edge_attr):
            h = self.in_norm(self.in_proj(x))
            for conv, norm in zip(self.convs, self.norms):
                h = norm(h + torch.relu(conv(h, edge_index, edge_attr)))   # residual + norm
            return h

    class EGATActorCritic(nn.Module):
        """Residual E-GAT actor-critic (proposal §4.4 decision module).

        Encoder: message passing over the road graph with edge features
        [t0, tpred, length, current rho] and node features [dist-to-dest, is-dest,
        out-rho] -> node embeddings that carry NETWORK-WIDE congestion context. It is
        re-encoded per vehicle as load builds, so congestion propagates across the
        graph (the global context the per-node MLP lacked). Decoder: scores each
        candidate neighbour from [h_cur, h_w, h_dest, local edge features]; the critic
        reads a graph-pooled state value.

        Built with the graph `g` (fixed topology -> edge_index/edge_static buffers).
        Duck-typed to DRLRoutingAgent (.act); the encode is cached per vehicle in
        rollout and recomputed (with grad) per vehicle in the PPO update.
        """

        NODE_DIM = 3
        EDGE_DIM = 4    # [t0, tpred, length, rho]

        def __init__(self, g, hidden=128, layers=3, heads=4, cand_dim=EDGE_FEATURE_DIM):
            super().__init__()
            edge_list = list(g.edges())
            ei = torch.tensor([[u for u, _ in edge_list],
                               [v for _, v in edge_list]], dtype=torch.long)
            es = torch.tensor([[g.edges[e]["t0"], g.edges[e]["tpred"], g.edges[e]["length"]]
                               for e in edge_list], dtype=torch.float)
            self.register_buffer("edge_index", ei)
            self.register_buffer("edge_static", es)
            self.encoder = _EGATEncoder(self.NODE_DIM, self.EDGE_DIM, hidden, layers, heads)
            self.cand_norm = nn.LayerNorm(cand_dim)
            self.actor = nn.Sequential(nn.Linear(3 * hidden + cand_dim, hidden),
                                       nn.ReLU(), nn.Linear(hidden, 1))
            self.critic = nn.Sequential(nn.Linear(3 * hidden, hidden),
                                        nn.ReLU(), nn.Linear(hidden, 1))
            self._cv, self._H = None, None       # rollout cache: (vehicle id, node embeddings)

        @property
        def device(self):
            return self.edge_static.device

        # ---- graph encode / candidate decode (inputs moved to the model's device) ----
        def encode(self, node_dyn, edge_rho):
            dev = self.device
            node_dyn = torch.as_tensor(node_dyn, dtype=torch.float32).to(dev)
            edge_rho = torch.as_tensor(edge_rho, dtype=torch.float32).to(dev)
            edge_attr = torch.cat([self.edge_static, edge_rho.unsqueeze(-1)], dim=-1)
            return self.encoder(node_dyn, self.edge_index, edge_attr)

        def decode(self, H, cur, dest, cand_ids, cand_feats):
            cand_ids = torch.as_tensor(cand_ids, dtype=torch.long).to(H.device)
            cand_feats = torch.as_tensor(cand_feats, dtype=torch.float32).to(H.device)
            hc, hd, hpool = H[cur], H[dest], H.mean(0)
            k = cand_feats.shape[0]
            ctx = torch.cat([hc, hd]).unsqueeze(0).expand(k, -1)          # [k, 2H]
            logits = self.actor(torch.cat([ctx, H[cand_ids],
                                           self.cand_norm(cand_feats)], dim=-1)).squeeze(-1)
            value = self.critic(torch.cat([hc, hd, hpool])).squeeze(-1)
            return logits, value

        # ---- rollout (encode cached per vehicle) ----
        def _rollout_H(self, obs):
            if obs.vehicle != self._cv:
                self._cv = obs.vehicle
                self._H = self.encode(torch.as_tensor(obs.node_dyn),
                                      torch.as_tensor(obs.edge_rho))
            return self._H

        @torch.no_grad()
        def act(self, obs, greedy=False):
            if len(obs.neighbors) == 1:
                return 0
            logits, _ = self.decode(self._rollout_H(obs), obs.node, obs.dest,
                                    torch.as_tensor(obs.neighbors),
                                    torch.as_tensor(obs.feats))
            if greedy:
                return int(torch.argmax(logits))
            return int(Categorical(logits=logits).sample())

        @torch.no_grad()
        def act_with_value(self, obs):
            """Rollout step for PPO: returns (action_idx, log_prob, value) as floats."""
            logits, value = self.decode(self._rollout_H(obs), obs.node, obs.dest,
                                        torch.as_tensor(obs.neighbors),
                                        torch.as_tensor(obs.feats))
            dist = Categorical(logits=logits)
            a = dist.sample()
            return int(a), float(dist.log_prob(a)), float(value)

        def evaluate(self, H, tr, action):
            """Re-evaluate a stored transition under a grad-enabled H -> logp, value, entropy."""
            logits, value = self.decode(H, tr["cur"], tr["dest"], tr["cands"], tr["cand_feats"])
            dist = Categorical(logits=logits)
            a = torch.as_tensor(action, device=logits.device)
            return dist.log_prob(a), value, dist.entropy()

        def reset_cache(self):
            self._cv, self._H = None, None

        def load(self, path):
            self.load_state_dict(torch.load(path, map_location="cpu"))
            self.eval()
            return self

        def save(self, path):
            torch.save(self.state_dict(), path)

    class PPOTrainer:
        """PPO (clipped surrogate, eq.5) for the E-GAT actor-critic on RoutingEnv.

        Transitions are grouped by vehicle: the graph is re-encoded once per vehicle
        (load is ~constant within a trip) in both rollout and update, so the encoder
        gets gradients without re-encoding per step. The update is mini-batched over
        vehicles (`mb_vehicles`) to bound memory. The per-vehicle encode is the heavy
        op; on the dense graph it is slow on CPU — this is where k-NN sparsification
        (config.KNN) or a GPU pays off.
        """

        def __init__(self, env, agent, lr=3e-4, clip_eps=0.2, gamma=0.99,
                     gae_lambda=0.95, value_coef=0.5, entropy_coef=0.01,
                     epochs=4, max_grad_norm=0.5, mb_vehicles=16):
            self.env, self.agent = env, agent
            self.clip_eps, self.gamma, self.lam = clip_eps, gamma, gae_lambda
            self.value_coef, self.entropy_coef = value_coef, entropy_coef
            self.epochs, self.max_grad_norm, self.mb_vehicles = epochs, max_grad_norm, mb_vehicles
            self.opt = torch.optim.Adam(agent.parameters(), lr=lr)

        def collect_episode(self):
            """One full pass over all vehicles -> list of transitions."""
            self.agent.reset_cache()
            traj, obs = [], self.env.reset()
            while not self.env.done and obs is not None:
                a, logp, value = self.agent.act_with_value(obs)
                tr = {"vehicle": obs.vehicle, "node_dyn": obs.node_dyn, "edge_rho": obs.edge_rho,
                      "cur": obs.node, "dest": obs.dest,
                      "cands": torch.as_tensor(obs.neighbors),
                      "cand_feats": torch.as_tensor(obs.feats),
                      "action": a, "logp": logp, "value": value}
                obs, reward, _, _ = self.env.step(a)
                tr["reward"] = reward
                traj.append(tr)
            return traj

        def _gae(self, traj):
            adv, gae, next_value = [0.0] * len(traj), 0.0, 0.0
            for t in reversed(range(len(traj))):
                delta = traj[t]["reward"] + self.gamma * next_value - traj[t]["value"]
                gae = delta + self.gamma * self.lam * gae
                adv[t] = gae
                next_value = traj[t]["value"]
            returns = [a + traj[t]["value"] for t, a in enumerate(adv)]
            return adv, returns

        def update(self, traj):
            if not traj:
                return {}
            adv, returns = self._gae(traj)
            adv_t = torch.tensor(adv, dtype=torch.float32)
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
            ret_t = torch.tensor(returns, dtype=torch.float32)
            old_logp = torch.tensor([tr["logp"] for tr in traj], dtype=torch.float32)
            # group transition indices by vehicle (encode once per vehicle, decode per step)
            veh_in, veh_idx = {}, {}
            for i, tr in enumerate(traj):
                vid = tr["vehicle"]
                if vid not in veh_in:
                    veh_in[vid] = (torch.as_tensor(tr["node_dyn"]), torch.as_tensor(tr["edge_rho"]))
                veh_idx.setdefault(vid, []).append(i)
            vids = list(veh_idx)
            stats = {}
            for _ in range(self.epochs):
                for s in torch.randperm(len(vids)).split(self.mb_vehicles):   # vehicle minibatches
                    idxs, logp, values, ent = [], [], [], []
                    for vid in (vids[j] for j in s.tolist()):
                        H = self.agent.encode(*veh_in[vid])
                        for i in veh_idx[vid]:
                            lp, v, e = self.agent.evaluate(H, traj[i], traj[i]["action"])
                            logp.append(lp); values.append(v); ent.append(e); idxs.append(i)
                    idx_t = torch.tensor(idxs)
                    logp = torch.stack(logp); values = torch.stack(values)
                    entropy = torch.stack(ent).mean()
                    dev = self.agent.device
                    a, r, ol = adv_t[idx_t].to(dev), ret_t[idx_t].to(dev), old_logp[idx_t].to(dev)
                    ratio = torch.exp(logp - ol)
                    clipped = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
                    policy_loss = -torch.min(ratio * a, clipped * a).mean()
                    value_loss = ((values - r) ** 2).mean()
                    loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                    self.opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                    self.opt.step()
                    stats = {"policy_loss": float(policy_loss),
                             "value_loss": float(value_loss), "entropy": float(entropy)}
            return stats

        def train(self, iterations=100, log_every=10):
            self.agent.train()
            for it in range(1, iterations + 1):
                traj = self.collect_episode()
                stats = self.update(traj)
                if it % log_every == 0:
                    ret = sum(tr["reward"] for tr in traj)
                    print(f"[PPO] iter {it:4d}  return {ret:10.3f}  "
                          f"pi_loss {stats.get('policy_loss', 0):.4f}  "
                          f"v_loss {stats.get('value_loss', 0):.4f}")
            self.agent.eval()
            return self.agent


def policy_drl(g, demand, agent):
    """Route every vehicle by rolling `agent` out in RoutingEnv (greedy).

    Same (g, demand) -> paths contract as the analytic policies, so it drops straight
    into run_compare. `agent` is anything implementing DRLRoutingAgent.act.
    """
    env = RoutingEnv(g, demand, use_penalty=True)
    obs = env.reset()
    while not env.done and obs is not None:
        obs, _, _, _ = env.step(agent.act(obs, greedy=True))
    return env.paths


def make_drl_agent(spec, g):
    """Build an agent from a run_compare --drl spec:
        'placeholder'/'oracle' -> GreedyOracleAgent (analytic, no torch/training)
        <path.pt>              -> trained EGATActorCritic (built on `g`) from checkpoint
    """
    if spec in (None, "", "placeholder", "oracle"):
        return GreedyOracleAgent()
    if not _TORCH_OK:
        raise ImportError("PyTorch + torch_geometric are required for the DRL agent.")
    return EGATActorCritic(g).load(spec)
