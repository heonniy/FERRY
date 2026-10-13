"""Expert-parallel communication layout for one MoE layer, built on the host.

Tokens are dispatched as token-to-rank packets: a token selecting several experts on
the same destination rank is sent once, together with those expert IDs and their
routing weights. The destination returns one partial sum per received packet.
"""
import numpy as np
import torch
from numba import njit


@njit(cache=True, nogil=True)
def build_layout(selected, destinations, n_local, rank, experts):
    world = selected.shape[0] // n_local
    topk = selected.shape[1]
    first = rank * n_local
    send_idx = np.empty(n_local * world, np.int64)
    send_ids = np.full((n_local * world, topk), -1, np.int64)
    send_counts = np.zeros(world, np.int64)
    scratch = np.empty(topk, np.int64)
    sent = 0
    for peer in range(world):
        for token in range(first, first + n_local):
            size = 0
            for j in range(topk):
                if destinations[token, j] == peer:
                    expert = selected[token, j]
                    pos = size
                    while pos > 0 and scratch[pos - 1] > expert:
                        scratch[pos] = scratch[pos - 1]
                        pos -= 1
                    scratch[pos] = expert
                    size += 1
            if size:
                send_idx[sent] = token - first
                send_ids[sent, :size] = scratch[:size]
                sent += 1
                send_counts[peer] += 1
    recv_ids = np.full((n_local * world, topk), -1, np.int64)
    recv_counts = np.zeros(world, np.int64)
    expert_counts = np.zeros(experts, np.int64)
    received = 0
    for peer in range(world):
        for token in range(peer * n_local, (peer + 1) * n_local):
            size = 0
            for j in range(topk):
                if destinations[token, j] == rank:
                    expert = selected[token, j]
                    pos = size
                    while pos > 0 and scratch[pos - 1] > expert:
                        scratch[pos] = scratch[pos - 1]
                        pos -= 1
                    scratch[pos] = expert
                    size += 1
            if size:
                for j in range(size):
                    recv_ids[received, j] = scratch[j]
                    expert_counts[scratch[j]] += 1
                received += 1
                recv_counts[peer] += 1
    starts = np.zeros(experts + 1, np.int64)
    for e in range(experts):
        starts[e + 1] = starts[e] + expert_counts[e]
    rows = np.empty(starts[-1], np.int64)
    cols = np.empty(starts[-1], np.int64)
    cursor = starts[:-1].copy()
    for row in range(received):
        for col in range(topk):
            expert = recv_ids[row, col]
            if expert < 0:
                break
            rows[cursor[expert]] = row
            cols[cursor[expert]] = col
            cursor[expert] += 1
    return send_idx[:sent], send_ids[:sent], send_counts, recv_counts, starts, rows, cols


def device_layout(selected, destinations, tokens_per_rank, rank, experts, slot_of):
    """Build the layout and move every index array to the GPU in one transfer."""
    send_idx, send_ids, send_counts, recv_counts, starts, rows, cols = build_layout(
        selected, destinations, tokens_per_rank, rank, experts)
    group_experts = np.flatnonzero(starts[1:] > starts[:-1])
    pieces = [send_idx, send_ids.reshape(-1)]
    for e in group_experts:
        pieces += [rows[starts[e]:starts[e + 1]], cols[starts[e]:starts[e + 1]]]
    storage = torch.from_numpy(np.concatenate(pieces)).to('cuda', non_blocking=False)
    views = storage.split([len(p) for p in pieces])
    groups = []
    for i, e in enumerate(group_experts):
        slot = int(slot_of[e])
        assert slot >= 0, 'expert scheduled on this rank is not cached here'
        groups.append((int(e), views[2 + 2 * i], views[3 + 2 * i], slot))
    return dict(send_idx=views[0], send_ids=views[1].view(-1, selected.shape[1]),
                send_counts=send_counts.tolist(), recv_counts=recv_counts.tolist(), groups=groups)
