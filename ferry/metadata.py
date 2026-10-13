"""Global routing metadata (paper Section 5.1 and Appendix A.1).

At every MoE layer each rank contributes the top-k expert IDs of its tokens (and, for the
routing-probability score, the router probabilities of its ceil(W / world) most recent
tokens). One all-gather gives every rank the global demand d_{r,e} and the history rows
of the W most recent routed tokens, taken evenly from all ranks, for cache eviction:
  frequency   : row = top-k selection indicator of the token (Eq. 13, default)
  probability : row = full router probability vector of the token (Eq. 12)
"""
import numpy as np
import torch
import torch.distributed as dist


class GateHistory:
    """Per-layer mean of the history rows of the W most recent routed tokens."""

    def __init__(self, layers, experts, window=128):
        self.window = window
        self.rows = np.zeros((layers, window, experts), np.float64)
        self.count = np.zeros(layers, np.int64)
        self.head = np.zeros(layers, np.int64)

    def update(self, layer, probs):
        w = self.window
        if len(probs) >= w:
            self.rows[layer] = probs[-w:]
            self.count[layer] = w
            self.head[layer] = 0
            return
        for row in probs:
            self.rows[layer, self.head[layer]] = row
            self.head[layer] = (self.head[layer] + 1) % w
            self.count[layer] = min(w, self.count[layer] + 1)

    def scores(self, layer):
        n = max(1, int(self.count[layer]))
        return (self.rows[layer].sum(0) / n).astype(np.float32)


class RouteGather:
    """Fixed-size all-gather of expert IDs (and the router-probability tail) per rank.

    Every rank must hold the same number of tokens in a forward pass, which holds
    for the fixed-length, fixed-batch workloads of the evaluation.
    """

    def __init__(self, topk, experts, window, score='frequency'):
        if score not in ('frequency', 'probability'):
            raise ValueError(score)
        self.topk, self.experts, self.window, self.score = topk, experts, window, score
        self.world, self.rank = dist.get_world_size(), dist.get_rank()
        self.buffers = {}

    def _buffers(self, tokens):
        if tokens not in self.buffers:
            tail = min(tokens, -(-self.window // self.world))
            ids = tokens * self.topk
            size = ids + (tail * self.experts * 4 if self.score == 'probability' else 0)
            self.buffers[tokens] = (tail, ids, size,
                                    torch.empty(size, dtype=torch.uint8, device='cuda'),
                                    torch.empty(self.world * size, dtype=torch.uint8, device='cuda'),
                                    torch.empty(self.world * size, dtype=torch.uint8, pin_memory=True),
                                    np.repeat(np.arange(self.world, dtype=np.int64), tokens))
        return self.buffers[tokens]

    def collect(self, selected, probs_tail):
        tokens = selected.shape[0]
        tail, ids, size, send, recv, host, origins = self._buffers(tokens)
        send[:ids].copy_(selected.to(torch.uint8).reshape(-1))
        if self.score == 'probability':
            send[ids:].copy_(probs_tail[-tail:].float().contiguous().view(torch.uint8).reshape(-1))
        dist.all_gather_into_tensor(recv, send)
        host.copy_(recv)
        raw = host.numpy().reshape(self.world, size)
        expert_ids = raw[:, :ids].reshape(self.world * tokens, self.topk).astype(np.int64)
        if self.score == 'probability':
            rows = raw[:, ids:].copy().view(np.float32).reshape(self.world * tail, self.experts)
        else:
            recent = expert_ids.reshape(self.world, tokens, self.topk)[:, tokens - tail:].reshape(-1, self.topk)
            rows = np.zeros((len(recent), self.experts), np.float64)
            np.put_along_axis(rows, recent, 1.0, axis=1)
        return expert_ids, origins, rows
