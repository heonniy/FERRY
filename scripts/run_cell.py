"""Run one workload cell with one placement policy (launch with torchrun, one process per GPU).

Example:
  CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node 4 scripts/run_cell.py \
      --model-path $MODEL_DIR/Qwen3-30B-A3B-Instruct-2507 --workloads $WORK_DIR/ShareGPT/WORKLOADS.json \
      --cell Qwen3_ShareGPT_R4_C30_B16_L512_O64 --policy ferry --topology topology/qwen3.json \
      --output results/Qwen3_ShareGPT_R4_C30_B16_L512_O64/ferry
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def pin_cpus(mode):
    """Give each rank a disjoint CPU set: the host CPUs split evenly by physical GPU index."""
    if mode == 'none':
        return None
    local = int(os.environ.get('LOCAL_RANK', '0'))
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    try:
        physical = int(visible.split(',')[local]) if visible else local
    except ValueError:      # GPU UUIDs: fall back to the local index
        physical = local
    total = len(subprocess.check_output(['nvidia-smi', '-L'], text=True).strip().splitlines())
    cpus = sorted(os.sched_getaffinity(0))
    per = len(cpus) // total
    mine = cpus[physical * per:(physical + 1) * per]
    if not mine:
        return None
    os.sched_setaffinity(0, mine)
    return mine


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def generate(model, rt, ids, steps):
    """Greedy decoding with a common release time; returns per-rank timestamps."""
    import torch
    import torch.distributed as dist
    dist.barrier()
    torch.cuda.synchronize()
    release = torch.tensor([time.perf_counter_ns() + 200_000_000 if dist.get_rank() == 0 else 0],
                           device='cuda', dtype=torch.int64)
    dist.broadcast(release, 0)
    start = int(release.item())
    while time.perf_counter_ns() < start:
        time.sleep(1e-4)
    mask = torch.ones_like(ids)
    past, tokens, stamps = None, [], []
    with torch.inference_mode():
        for step in range(steps):
            position = torch.arange(mask.shape[1] - ids.shape[1], mask.shape[1], device='cuda')
            out = model(input_ids=ids, attention_mask=mask, position_ids=position[None].expand(len(ids), -1),
                        past_key_values=past, use_cache=True, logits_to_keep=1)
            past = out.past_key_values
            next_ids = out.logits[:, -1].argmax(-1)
            tokens.append(next_ids)
            torch.cuda.synchronize()
            stamps.append(time.perf_counter_ns())
            if step < steps - 1:
                ids = next_ids[:, None]
                mask = torch.cat((mask, mask.new_ones((len(ids), 1))), 1)
    return dict(release_ns=start, first_ns=stamps[0], end_ns=stamps[-1],
                tokens=torch.stack(tokens, 1).cpu().tolist())


def main(a):
    cpus = pin_cpus(a.cpu_affinity)
    import numpy as np
    import torch
    import torch.distributed as dist
    from ferry.model import install_hooks, load, read_spec
    from ferry.quota import Topology
    from ferry.runtime import OffloadRuntime

    local = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local)
    torch.set_num_threads(2)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group('nccl', device_id=torch.device('cuda', local))
    rank, world = dist.get_rank(), dist.get_world_size()

    manifest = Path(a.workloads).resolve()
    cell = next((c for c in json.loads(manifest.read_text())['cells'] if c['cell'] == a.cell), None)
    if cell is None:
        raise SystemExit(f'cell {a.cell} is not in {manifest}; rebuild it with --world {world}')
    assert cell['world'] == world, f"cell {a.cell} is defined for {cell['world']} GPUs"
    spec = read_spec(a.model_path)
    ratio = a.cache_ratio if a.cache_ratio is not None else cell['cache_percent'] / 100
    capacities = spec.cache_slots(ratio, world)
    if a.policy == 'ferry' and a.topology is None:
        raise SystemExit('--topology is required for FERRY (see scripts/profile_h2d.py); '
                         'use --topology uniform to assume equal H2D bandwidth on every GPU')
    topology = (Topology.uniform(world, spec.experts) if a.topology in (None, 'uniform')
                else Topology(a.topology, world, spec.experts))
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if topology.gpus and visible and [str(g) for g in topology.gpus] != visible.split(','):
        raise SystemExit(f'topology was profiled on GPUs {topology.gpus}, running on {visible}')
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)

    model, pool = load(a.model_path, spec)
    rt = OffloadRuntime(spec, pool, capacities, a.policy, topology, a.history_window, a.balance_tol_bps,
                        a.salt, a.eviction_score)
    install_hooks(model, spec, rt, a.history_window)
    batch = cell['local_batch']
    steps = 2 if a.smoke else cell['output_tokens']
    repeats = 1 if a.smoke else a.repeats
    discard = 0 if a.smoke else a.discard
    # Run 0 uses the warmup requests. The next `discard` runs repeat the target requests
    # but are not reported: the first pass over new shapes is slow (allocator, NCCL buffers).
    rows = []
    for repeat in range(1 + discard + repeats):
        phase = 'warmup' if repeat == 0 else 'discard' if repeat <= discard else 'target'
        source = 'warmup' if repeat == 0 else 'target'
        requests = json.loads((manifest.parent / cell[source]).read_text())['requests']
        local_requests = requests[rank * batch:(rank + 1) * batch]
        assert all(r['origin_rank'] == rank for r in local_requests)
        ids = torch.tensor([r['input_ids'][-32:] if a.smoke else r['input_ids'] for r in local_requests],
                           device='cuda')
        rt.reset()
        torch.cuda.reset_peak_memory_stats()
        result = generate(model, rt, ids, steps)
        rt.h2d.synchronize()
        result.update(rank=rank, repeat=repeat, phase=phase, cpus=cpus,
                      fetches_per_rank=rt.fetches_per_rank.tolist(),
                      decode_fetches_per_rank=rt.decode_fetches_per_rank.tolist(),
                      h2d_copies=rt.h2d.copies, serial_wait_s=rt.serial_wait_ns / 1e9,
                      dispatch_bytes=rt.transport.dispatch_bytes, combine_bytes=rt.transport.combine_bytes,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated())
        write(out / f'repeat{repeat}_rank{rank}.json', result)
        dist.barrier()
        if rank == 0:
            ranks = [json.loads((out / f'repeat{repeat}_rank{r}.json').read_text()) for r in range(world)]
            start = ranks[0]['release_ns']
            first = max(x['first_ns'] for x in ranks)
            end = max(x['end_ns'] for x in ranks)
            e2e = (end - start) / 1e9
            row = dict(repeat=repeat, phase=phase, TTFT=(first - start) / 1e9,
                       TPOT=(end - first) / 1e9 / (steps - 1), E2E=e2e,
                       TPS=world * batch * steps / e2e, global_requests=world * batch,
                       output_tokens=steps, decode_fetches_per_rank=ranks[0]['decode_fetches_per_rank'])
            write(out / f'repeat{repeat}.json', row)
            print(json.dumps(row), flush=True)
            if phase == 'target':
                rows.append(row)
        dist.barrier()
    rt.close()
    if rank == 0:
        write(out / 'result.json', dict(
            status='PASS', cell=a.cell, policy=a.policy, model_type=spec.model_type, world=world,
            cache_ratio=ratio, cache_slots_per_rank=capacities, history_window=a.history_window,
            eviction_score=a.eviction_score,
            balance_tol_bps=a.balance_tol_bps, salt=a.salt, discarded_runs=discard,
            topology=a.topology, smoke=a.smoke, repeats=rows))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model-path', required=True)
    p.add_argument('--workloads', required=True, help='WORKLOADS.json from scripts/prepare_workloads.py')
    p.add_argument('--cell', required=True)
    p.add_argument('--policy', choices=('ferry', 'random'), required=True)
    p.add_argument('--topology', help="topology JSON from scripts/build_quota_lut.py, or 'uniform'")
    p.add_argument('--output', required=True)
    p.add_argument('--repeats', type=int, default=3, help='measured runs after one warmup run')
    p.add_argument('--discard', type=int, default=1,
                   help='unreported runs on the target requests before the measured ones')
    p.add_argument('--cache-ratio', type=float, help='override the cell expert-cache budget (fraction)')
    p.add_argument('--history-window', type=int, default=128, help='W: routed tokens in the eviction history')
    p.add_argument('--eviction-score', choices=('frequency', 'probability'), default='frequency',
                   help='selection frequency (Eq. 13) or mean router probability (Eq. 12) over the last W tokens')
    p.add_argument('--balance-tol-bps', type=int, default=200,
                   help='Step 3 balance tolerance in basis points (0 = strict argmin)')
    p.add_argument('--salt', type=int, default=1, help='random seed salt for the random policy')
    p.add_argument('--cpu-affinity', choices=('auto', 'none'), default='auto')
    p.add_argument('--smoke', action='store_true', help='32-token inputs, 2 output tokens, 1 repeat')
    main(p.parse_args())
