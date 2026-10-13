"""Collect results/<cell>/<policy>/result.json into the main table (mean ± sample std).

  python scripts/summarize.py --results results --output results/main_table.md
"""
import argparse
import json
import statistics
from pathlib import Path

NAMES = {'random': 'FERRY-Random', 'ferry': 'FERRY (Ours)'}


def fmt(values, digits):
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return f'{mean:.{digits}f} ± {std:.{digits}f}'


def main(a):
    rows = []
    for path in sorted(Path(a.results).glob('*/*/result.json')):
        r = json.loads(path.read_text())
        if r.get('status') != 'PASS' or r.get('smoke'):
            continue
        reps = r['repeats']
        model, dataset, _, _, batch, length, _ = r['cell'].split('_')
        rows.append((model, dataset, int(length[1:]), int(batch[1:]), r['policy'],
                     fmt([x['TPS'] for x in reps], 1), fmt([x['TTFT'] for x in reps], 2),
                     fmt([x['TPOT'] for x in reps], 3), len(reps)))
    order = {'random': 0, 'ferry': 1}
    rows.sort(key=lambda x: (x[0], x[1], x[2], x[3], order.get(x[4], 9)))
    lines = ['| Model | Dataset | Input | Batch/GPU | Method | TPS (token/s) ↑ | TTFT (s) ↓ | TPOT (s) ↓ | n |',
             '|---|---|---:|---:|---|---|---|---|---:|']
    lines += [f'| {m} | {d} | {l} | {b} | {NAMES.get(p, p)} | {tps} | {ttft} | {tpot} | {n} |'
              for m, d, l, b, p, tps, ttft, tpot, n in rows]
    text = '\n'.join(lines) + '\n'
    print(text)
    if a.output:
        Path(a.output).write_text(text)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--results', default='results')
    p.add_argument('--output')
    main(p.parse_args())
