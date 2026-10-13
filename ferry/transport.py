"""Dispatch and combine all-to-all exchanges (BF16 wire format).

Packet packing, partial reduction and the final scatter-add run in C++ (ferry/csrc);
only the all-to-all collectives are issued from Python.
"""
import torch
import torch.distributed as dist


class Transport:
    def __init__(self, ops, topk, experts):
        self.ops, self.topk, self.experts = ops, topk, experts
        self.rank = dist.get_rank()
        self.dispatch_bytes = self.combine_bytes = 0

    @staticmethod
    def _exchange(x, send, recv):
        y = torch.empty((sum(recv),) + tuple(x.shape[1:]), device=x.device, dtype=x.dtype)
        dist.all_to_all_single(y, x, output_split_sizes=recv, input_split_sizes=send)
        return y

    def dispatch(self, hidden, selected, weights, layout):
        """Send each packet as [hidden | per-expert routing weights | expert IDs] bytes."""
        payload = self.ops.pack_dispatch(hidden, selected, weights, layout['send_idx'],
                                         layout['send_ids'], self.experts)
        send, recv = layout['send_counts'], layout['recv_counts']
        raw = self._exchange(payload, send, recv)
        self.dispatch_bytes += (sum(send) - send[self.rank]) * payload.shape[1]
        return self.ops.unpack_dispatch(raw, hidden.shape[1], self.topk)

    def combine(self, hidden, parts, layout):
        """Sum expert outputs into one partial per received packet and send them back."""
        recv, send = layout['recv_counts'], layout['send_counts']
        partial = self.ops.reduce_partials(parts, [g[1] for g in layout['groups']], sum(recv),
                                           hidden.shape[1], hidden)
        returned = self._exchange(partial, recv, send)
        self.combine_bytes += (sum(recv) - recv[self.rank]) * hidden.shape[1] * 2
        return self.ops.scatter_returned(returned, layout['send_idx'], hidden.shape[0])
