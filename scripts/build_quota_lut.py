"""Turn an H2D profile into bandwidth domains and the Step 2 quota lookup table.

Domains: every GPU is a singleton domain. Shared domains are either every profiled
GPU subset (the default) or explicit groups such as PCIe switches or
NUMA nodes (--groups 0,1:2,3). For each number of missing experts m = 0..E the table
stores the quota of Algorithm 1 (lines 1-5) without capacity limits.

  python scripts/build_quota_lut.py --profile topology/qwen3_profile.json \
      --model-path $MODEL_DIR/Qwen3-30B-A3B-Instruct-2507 --output topology/qwen3.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ferry.model import read_spec  # noqa: E402
from ferry.quota import build_lut  # noqa: E402


def main(a):
    profile = json.loads(Path(a.profile).read_text())
    spec = read_spec(a.model_path)
    if profile['expert_bytes'] != spec.expert_bytes:
        print(f"warning: profiled with {profile['expert_bytes']} B copies, model experts are "
              f"{spec.expert_bytes} B", file=sys.stderr)
    world = profile['world']
    by_mask = {d['mask']: d for d in profile['domains']}
    if a.groups:
        masks = [1 << r for r in range(world)]
        masks += [sum(1 << int(g) for g in group.split(',')) for group in a.groups.split(':')]
    else:
        masks = sorted(by_mask, key=lambda m: (bin(m).count('1'), m))
    missing = [m for m in masks if m not in by_mask]
    if missing:
        raise SystemExit(f'profile has no measurement for GPU sets {missing}')
    domains = [dict(gpus=by_mask[m]['gpus'], mask=m, gib_per_s=by_mask[m]['gib_per_s']) for m in masks]
    lut = build_lut(spec.experts, world, [d['mask'] for d in domains],
                    [1.0 / d['gib_per_s'] for d in domains])
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(world=world, physical_gpus=profile['physical_gpus'],
                                   expert_bytes=profile['expert_bytes'], model_type=spec.model_type,
                                   domains=domains, quota_lut=lut.tolist())) + '\n')
    for m in sorted({1, world, spec.experts // 4, spec.experts // 2, spec.experts}):
        print(f'm={m:4d}: quota {lut[m].tolist()}')
    print(f'wrote {out}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--profile', required=True)
    p.add_argument('--model-path', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--groups', help='explicit shared domains, e.g. 0,1:2,3 (default: every profiled subset)')
    main(p.parse_args())
