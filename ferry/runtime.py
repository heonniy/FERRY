"""Multi-GPU offloading runtime: DP attention + EP routed experts with a shared GPU cache.

Per MoE layer, on every rank:
  1. gather routing metadata (expert IDs, router-probability tail) from all ranks;
  2. update the routing-score history and plan placement (identical on every rank);
  3. enqueue this rank's H2D fetches of the experts it must load;
  4. dispatch token packets to the ranks that execute their experts (all-to-all);
  5. wait until this rank's fetches of the layer have completed (serial execution:
     expert GEMMs never overlap H2D transfers);
  6. run the routed experts held by this rank and return partial sums (all-to-all).
"""
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from .layout import device_layout
from .metadata import GateHistory, RouteGather
from .policy import Controller, layer_slots
from .transport import Transport


@lru_cache(None)
def expert_ops():
    from torch.utils.cpp_extension import load
    root = Path(__file__).parent / 'csrc'
    return load(name='ferry_expert_ops', sources=[str(root / 'expert_ops.cpp'), str(root / 'expert_ops.cu')],
                extra_cflags=['-O2'], extra_cuda_cflags=['-O2'], verbose=False)


class OffloadRuntime:
    def __init__(self, spec, pool, capacities, policy, topology, history_window=128,
                 balance_tol_bps=200, salt=1, eviction_score='frequency'):
        self.spec = spec
        self.pool = pool
        self.rank, self.world = dist.get_rank(), dist.get_world_size()
        self.capacities = capacities
        self.cap = capacities[self.rank]
        self.window = history_window
        self.ops = expert_ops()
        self.cache = torch.empty((self.cap, spec.expert_elements), dtype=torch.bfloat16, device='cuda')
        self.controller = Controller(spec, capacities, policy, topology, balance_tol_bps, salt)
        self.gather = RouteGather(spec.topk, spec.experts, history_window, eviction_score)
        self.transport = Transport(self.ops, spec.topk, spec.experts)
        self.h2d = None
        self.reset()

    def reset(self):
        """Empty expert cache and routing history (start of every measured run)."""
        if self.h2d is not None:
            self.h2d.synchronize()
        torch.cuda.synchronize()
        self.controller.reset()
        self.history = GateHistory(self.spec.layers, self.spec.experts, self.window)
        self.h2d = self.ops.H2DEngine(self.cache, self.pool)
        self.event = 0
        self.fetches_per_rank = np.zeros(self.world, np.int64)
        self.decode_fetches_per_rank = np.zeros(self.world, np.int64)
        self.serial_wait_ns = 0

    def execute(self, layer, hidden, selected, weights, probs_tail):
        spec = self.spec
        ids, origins, tail = self.gather.collect(selected, probs_tail)
        self.history.update(layer, tail)
        fetches, destinations, counts = self.controller.plan(
            self.event, layer, ids, origins, self.history.scores(layer))
        self.fetches_per_rank += counts
        if self.event >= spec.layers:
            self.decode_fetches_per_rank += counts
        mine = fetches[fetches[:, 0] == self.rank]
        fetch_slots = mine[:, 2].tolist()
        self.h2d.fetch(fetch_slots, mine[:, 1].tolist())
        slot_of = layer_slots(self.controller.slots[self.rank], self.cap, layer, spec.experts)
        layout = device_layout(ids, destinations, selected.shape[0], self.rank, spec.experts, slot_of)
        received, received_weights = self.transport.dispatch(hidden, selected, weights, layout)
        started = time.perf_counter_ns()
        self.h2d.wait_host(fetch_slots)
        self.serial_wait_ns += time.perf_counter_ns() - started
        groups = layout['groups']
        slots = [g[3] for g in groups]
        parts = self.ops.execute_experts(self.cache, received, received_weights, slots,
                                         [g[1] for g in groups], [g[2] for g in groups],
                                         spec.hidden, spec.intermediate)
        self.h2d.mark_used(slots)
        self.event += 1
        return self.transport.combine(hidden, parts, layout)

    def close(self):
        if self.h2d is not None:
            self.h2d.synchronize()
