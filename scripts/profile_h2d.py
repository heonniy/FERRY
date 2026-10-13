"""Offline H2D bandwidth profiling for topology-aware allocation (paper Section 5.2).

For every non-empty subset of the visible GPUs, all GPUs of the subset copy expert-sized
buffers from their own pinned host buffer at the same time. The per-GPU effective
bandwidth and the subset's aggregate bandwidth are recorded. Singleton subsets give the
individual GPU bandwidth; larger subsets expose shared PCIe-switch / NUMA / host limits.

Run it on the same GPUs (same CUDA_VISIBLE_DEVICES order) as the evaluation, with the
machine otherwise idle:
  CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/profile_h2d.py \
      --model-path $MODEL_DIR/Qwen3-30B-A3B-Instruct-2507 --output topology/qwen3_profile.json
"""
import argparse
import itertools
import json
import os
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ferry.model import read_spec  # noqa: E402


def measure(subset, sources, targets, streams, copies, spin_cycles):
    starts, ends = {}, {}
    for g in subset:
        with torch.cuda.device(g), torch.cuda.stream(streams[g]):
            torch.cuda._sleep(spin_cycles)          # hold the stream until every GPU is enqueued
            starts[g] = torch.cuda.Event(enable_timing=True)
            ends[g] = torch.cuda.Event(enable_timing=True)
            starts[g].record(streams[g])
    for i in range(copies):
        for g in subset:
            with torch.cuda.device(g), torch.cuda.stream(streams[g]):
                targets[g][i % len(targets[g])].copy_(sources[g], non_blocking=True)
    for g in subset:
        with torch.cuda.device(g):
            ends[g].record(streams[g])
    for g in subset:
        ends[g].synchronize()
    nbytes = sources[subset[0]].numel() * sources[subset[0]].element_size()
    return {g: copies * nbytes / (starts[g].elapsed_time(ends[g]) / 1e3) / 2**30 for g in subset}


def main(a):
    spec = read_spec(a.model_path)
    world = torch.cuda.device_count()
    gpus = list(range(world))
    elements = spec.expert_elements
    sources, targets, streams = {}, {}, {}
    for g in gpus:
        with torch.cuda.device(g):
            sources[g] = torch.randn(elements, dtype=torch.bfloat16).pin_memory()
            targets[g] = [torch.empty(elements, dtype=torch.bfloat16, device=g) for _ in range(4)]
            streams[g] = torch.cuda.Stream(device=g)
    subsets = [s for n in range(1, world + 1) for s in itertools.combinations(gpus, n)]
    for g in gpus:      # warm up every copy engine
        measure((g,), sources, targets, streams, 8, 0)
    samples = {s: [] for s in subsets}
    for repeat in range(a.repeats):
        order = subsets if repeat % 2 == 0 else subsets[::-1]
        for s in order:
            samples[s].append(measure(s, sources, targets, streams, a.copies, a.spin_cycles))
    domains = []
    for s in subsets:
        per_gpu = {g: statistics.median(x[g] for x in samples[s]) for g in s}
        domains.append(dict(gpus=list(s), mask=sum(1 << g for g in s),
                            gib_per_s=sum(per_gpu.values()),
                            per_gpu_gib_per_s=[per_gpu[g] for g in s]))
        if len(s) == 1 or len(s) == world:
            print(f'GPUs {list(s)}: aggregate {domains[-1]["gib_per_s"]:.2f} GiB/s, per GPU '
                  + ', '.join(f'{v:.2f}' for v in domains[-1]['per_gpu_gib_per_s']))
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(world=world, physical_gpus=visible.split(',') if visible else gpus,
                                   expert_bytes=spec.expert_bytes, copies=a.copies, repeats=a.repeats,
                                   domains=domains), indent=1) + '\n')
    print(f'wrote {out}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model-path', required=True, help='used only to size the copies like one expert')
    p.add_argument('--output', required=True)
    p.add_argument('--copies', type=int, default=64, help='back-to-back copies per GPU per sample')
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--spin-cycles', type=int, default=100_000_000,
                   help='GPU spin before the copies so that all GPUs start together')
    main(p.parse_args())
