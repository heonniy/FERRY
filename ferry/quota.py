"""Step 2: topology-aware expert allocation (paper Eq. 6, 8 and Algorithm 1, lines 1-5).

A bandwidth domain g is a set of ranks sharing an effective H2D bandwidth B_g (a single
GPU link, a PCIe switch, a NUMA node, ...). With q_r experts fetched by rank r,

    T_fetch(q) = S * max_g  sum_{r in g} q_r / B_g .

The allocator starts from q = 0 and, m times, increments the quota of the rank whose
increment yields the smallest T_fetch, subject to q_r < u_r. Ties go to the rank with
the smaller summed domain time, then the smaller individual transfer time, then the
lower rank index.

Domains and bandwidths are profiled offline (scripts/profile_h2d.py). Because S is a
common factor, the quota rows depend only on m when no capacity limit binds; they are
precomputed into a lookup table (scripts/build_quota_lut.py) and the greedy is only
re-run at runtime when a table row would exceed some rank's admission capacity u_r.
"""
import json
from pathlib import Path

import numpy as np
from numba import njit


@njit(cache=True)
def greedy_quota(m, capacity, masks, seconds):
    """masks[g]: bitmask of ranks in domain g; seconds[g] = 1 / B_g (time per expert)."""
    world = len(capacity)
    quota = np.zeros(world, np.int64)
    load = np.zeros(len(masks), np.int64)
    single = np.zeros(world, np.float64)
    for g in range(len(masks)):
        for r in range(world):
            if masks[g] == (1 << r):
                single[r] = seconds[g]
    for _ in range(m):
        best = -1
        best_t = np.inf
        best_sum = np.inf
        best_own = np.inf
        for r in range(world):
            if quota[r] >= capacity[r]:
                continue
            t = 0.0
            total = 0.0
            for g in range(len(masks)):
                v = (load[g] + ((masks[g] >> r) & 1)) * seconds[g]
                total += v
                if v > t:
                    t = v
            own = (quota[r] + 1) * single[r]
            key_better = False
            if t < best_t * (1 - 1e-12):
                key_better = True
            elif t <= best_t * (1 + 1e-12):
                # Equal makespan: prefer the smaller total domain time, then the faster GPU.
                if total < best_sum * (1 - 1e-12):
                    key_better = True
                elif total <= best_sum * (1 + 1e-12) and own < best_own * (1 - 1e-12):
                    key_better = True
            if key_better:
                best, best_t, best_sum, best_own = r, t, total, own
        assert best >= 0, 'missing experts exceed the total admission capacity'
        quota[best] += 1
        for g in range(len(masks)):
            load[g] += (masks[g] >> best) & 1
    return quota


def build_lut(max_misses, world, masks, seconds):
    unlimited = np.full(world, 1 << 30, np.int64)
    masks = np.asarray(masks, np.int64)
    seconds = np.asarray(seconds, np.float64)
    return np.stack([greedy_quota(m, unlimited, masks, seconds) for m in range(max_misses + 1)])


class Topology:
    """Profiled bandwidth domains plus the precomputed quota lookup table."""

    def __init__(self, path, world, experts):
        data = json.loads(Path(path).read_text())
        if data['world'] != world:
            raise ValueError(f"topology file is for {data['world']} GPUs, runtime has {world}")
        self.masks = np.asarray([d['mask'] for d in data['domains']], np.int64)
        self.seconds = np.asarray([1.0 / d['gib_per_s'] for d in data['domains']], np.float64)
        self.lut = np.asarray(data['quota_lut'], np.int64)
        if self.lut.shape[0] < experts + 1 or self.lut.shape[1] != world:
            raise ValueError('quota lookup table does not cover this model; rebuild it')
        assert all(int(self.lut[m].sum()) == m for m in range(len(self.lut)))
        self.gpus = data.get('physical_gpus')

    @classmethod
    def uniform(cls, world, experts):
        """Equal singleton bandwidths and no shared domain: Step 2 is an even split."""
        self = cls.__new__(cls)
        self.masks = np.asarray([1 << r for r in range(world)], np.int64)
        self.seconds = np.ones(world, np.float64)
        self.lut = build_lut(experts, world, self.masks, self.seconds)
        self.gpus = None
        return self
