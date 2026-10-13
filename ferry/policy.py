"""Global expert-placement controller (paper Section 5, Steps 1-4).

Every rank holds an identical copy of this state and receives the same gathered
routing metadata, so every rank derives the same placement without extra messages.

Cache state: slots[r, i] holds the expert key (layer * E + expert) cached in slot i of
rank r, or -1. owner[key] is the rank caching the key, or -1. Each expert is cached on
at most one rank and executes there.

Policies
  FERRY  : Step 2 topology-aware quota (ferry.quota) + Step 3 load-balanced assignment.
  RANDOM : every missing expert goes to an i.i.d. uniformly random rank (no quota, no
           load balance); the reference used for FERRY-Random.
Both use the same Step 4 routing-aware eviction.
"""
import numpy as np
from numba import njit

from .quota import greedy_quota

FERRY, RANDOM = 0, 1
POLICIES = {'ferry': FERRY, 'random': RANDOM}


@njit(cache=True)
def _demand(selected, origins, experts, world):
    """d[e, r]: number of tokens from rank r that select expert e (Step 1)."""
    d = np.zeros((experts, world), np.int64)
    for t in range(selected.shape[0]):
        r = origins[t]
        for k in range(selected.shape[1]):
            d[selected[t, k], r] += 1
    return d


@njit(cache=True)
def _admission_capacity(slots, capacities, layer, active, experts):
    """u_r: empty slots plus slots whose expert is not needed by the current layer."""
    world = len(capacities)
    u = np.zeros(world, np.int64)
    for r in range(world):
        for i in range(capacities[r]):
            key = slots[r, i]
            if key < 0 or not (key // experts == layer and active[key % experts]):
                u[r] += 1
    return u


@njit(cache=True)
def load_balanced_assignment(demand, misses, owner, layer, quota, balance_tol_bps):
    """Step 3 (Algorithm 1, lines 6-14).

    N_r starts from the tokens routed to experts already cached on r. Missing experts
    are processed in descending order of d_e. For each, the projected peak load is
    P_r = max(max_j N_j, N_r + d_e) over ranks with quota left. Ranks whose P_r is
    within balance_tol_bps basis points of the minimum form the balanced set R_e; among
    them the expert goes to the rank holding most of its tokens (max d_{r,e}), ties
    broken by smaller P_r, smaller N_r + d_e, then rank index. balance_tol_bps = 0 is the
    strict rule of the paper (R_e = argmin P_r).
    """
    experts, world = demand.shape
    m = len(misses)
    loads = np.zeros(world, np.int64)
    base = layer * experts
    for e in range(experts):
        r = owner[base + e]
        if r >= 0:
            loads[r] += demand[e].sum()
    totals = np.empty(m, np.int64)
    for i in range(m):
        totals[i] = demand[misses[i]].sum()
    order = np.argsort(-totals, kind='mergesort')
    remaining = quota.copy()
    answer = np.empty(m, np.int64)
    for oi in range(m):
        i = order[oi]
        e = misses[i]
        total = totals[i]
        peak = loads.max()
        projected = np.full(world, 1 << 60, np.int64)
        lowest = 1 << 60
        for r in range(world):
            if remaining[r] > 0:
                projected[r] = max(peak, loads[r] + total)
                lowest = min(lowest, projected[r])
        limit = lowest + (max(1, lowest * balance_tol_bps // 10000) if balance_tol_bps > 0 else 0)
        best = -1
        for r in range(world):
            if remaining[r] <= 0 or projected[r] > limit:
                continue
            if best < 0:
                best = r
                continue
            a, b = demand[e, r], demand[e, best]
            if a != b:
                if a > b:
                    best = r
            elif projected[r] != projected[best]:
                if projected[r] < projected[best]:
                    best = r
            elif loads[r] < loads[best]:
                best = r
        answer[i] = best
        remaining[best] -= 1
        loads[best] += total
    return answer


@njit(cache=True)
def _choose_slot(rank, layer, active, slots, capacities, last, scores, experts):
    """Step 4: an empty slot, else the lowest eviction score, then least-recently used."""
    best = -1
    best_score = np.inf
    best_used = 1 << 60
    best_key = 1 << 60
    for i in range(capacities[rank]):
        key = slots[rank, i]
        if key < 0:
            return i
        if key // experts == layer and active[key % experts]:
            continue
        score = scores[key // experts, key % experts]
        used = last[key]
        if (score < best_score or (score == best_score and
                (used < best_used or (used == best_used and key < best_key)))):
            best, best_score, best_used, best_key = i, score, used, key
    return best


@njit(cache=True)
def plan_layer(event, layer, selected, origins, gate_scores, policy, slots, capacities, owner,
               last, scores, quota_lut, masks, seconds, balance_tol_bps, salt):
    """Plan one MoE layer invocation and update the cache state in place.

    Returns fetches[n, 3] = (rank, key, slot), the per-route destination ranks, and
    the per-rank fetch counts.
    """
    experts = scores.shape[1]
    world = len(capacities)
    tick = event + 1
    scores[layer] = gate_scores
    demand = _demand(selected, origins, experts, world)
    active = np.zeros(experts, np.bool_)
    for e in range(experts):
        active[e] = demand[e].sum() > 0
    base = layer * experts
    misses = np.empty(experts, np.int64)
    m = 0
    for e in range(experts):
        if active[e] and owner[base + e] < 0:
            misses[m] = e
            m += 1
    misses = misses[:m]
    capacity = _admission_capacity(slots, capacities, layer, active, experts)
    if policy == FERRY:
        quota = quota_lut[m].copy()
        if np.any(quota > capacity):
            quota = greedy_quota(m, capacity, masks, seconds)
        assignment = load_balanced_assignment(demand, misses, owner, layer, quota, balance_tol_bps)
    else:
        # Stateless per-event draw, identical on every rank.
        np.random.seed(salt * 1000003 + event)
        left = capacity.copy()
        assignment = np.empty(m, np.int64)
        for i in range(m):
            r = np.random.randint(0, world)
            while left[r] <= 0:
                r = np.random.randint(0, world)
            assignment[i] = r
            left[r] -= 1
    fetches = np.empty((m, 3), np.int64)
    counts = np.zeros(world, np.int64)
    for i in range(m):
        r = assignment[i]
        key = base + misses[i]
        slot = _choose_slot(r, layer, active, slots, capacities, last, scores, experts)
        assert slot >= 0
        victim = slots[r, slot]
        if victim >= 0:
            owner[victim] = -1
        slots[r, slot] = key
        owner[key] = r
        fetches[i, 0] = r
        fetches[i, 1] = key
        fetches[i, 2] = slot
        counts[r] += 1
    for e in range(experts):
        if active[e]:
            last[base + e] = tick
    destinations = np.empty(selected.shape, np.int64)
    for t in range(selected.shape[0]):
        for k in range(selected.shape[1]):
            destinations[t, k] = owner[base + selected[t, k]]
    return fetches, destinations, counts


@njit(cache=True)
def layer_slots(slots_row, capacity, layer, experts):
    """Slot index of every expert of `layer` cached on this rank, or -1."""
    lookup = np.full(experts, -1, np.int64)
    for i in range(capacity):
        key = slots_row[i]
        if key >= 0 and key // experts == layer:
            lookup[key % experts] = i
    return lookup


class Controller:
    def __init__(self, spec, capacities, policy, topology, balance_tol_bps=200, salt=1):
        self.spec = spec
        self.policy = POLICIES[policy]
        self.capacities = np.asarray(capacities, np.int64)
        self.balance_tol_bps = int(balance_tol_bps)
        self.salt = int(salt)
        self.masks, self.seconds, self.lut = topology.masks, topology.seconds, topology.lut
        self.reset()

    def reset(self):
        world, layers, experts = len(self.capacities), self.spec.layers, self.spec.experts
        self.slots = np.full((world, int(self.capacities.max())), -1, np.int64)
        self.owner = np.full(layers * experts, -1, np.int64)
        self.last = np.zeros(layers * experts, np.int64)
        self.scores = np.zeros((layers, experts), np.float32)

    def plan(self, event, layer, selected, origins, gate_scores):
        return plan_layer(event, layer, selected, origins, gate_scores, self.policy, self.slots,
                          self.capacities, self.owner, self.last, self.scores, self.lut,
                          self.masks, self.seconds, self.balance_tol_bps, self.salt)
