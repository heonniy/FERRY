"""Build fixed, tokenized request manifests for the evaluation workloads.

Selection (deterministic, no sampling seed): scan the dataset in file order and keep the
first distinct conversations that end with a non-empty user turn and render to at least
max(--lengths) tokens under the chat template of every model. For every cell the target
requests are the first world*batch conversations; the warmup requests are a disjoint set
taken from the second half. Each request keeps the last L tokens of its rendered prompt.

Example:
  python scripts/prepare_workloads.py --dataset ShareGPT \
      --source $DATA_DIR/ShareGPT/ShareGPT_V3_unfiltered_cleaned_split.json \
      --model Qwen3=$MODEL_DIR/Qwen3-30B-A3B-Instruct-2507 \
      --model DeepSeekV2Lite=$MODEL_DIR/DeepSeek-V2-Lite-Chat --output workloads/ShareGPT
"""
import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def final_user_prefix(conversation, style):
    """Messages up to and including the last non-empty user turn of a well-formed prefix."""
    prefix, best = [], None
    for item in conversation or []:
        if not isinstance(item, dict):
            break
        if style == 'sharegpt':
            role = {'human': 'user', 'gpt': 'assistant'}.get(item.get('from'))
            content = item.get('value')
        else:
            role, content = item.get('role'), item.get('content')
        if role not in ('user', 'assistant') or not isinstance(content, str):
            break
        prefix.append({'role': role, 'content': content})
        if role == 'user' and content.strip():
            best = prefix.copy()
    return best


def source_rows(dataset, source):
    if dataset == 'ShareGPT':
        for index, row in enumerate(json.loads(Path(source).read_text())):
            yield index, row.get('id'), final_user_prefix(row.get('conversations'), 'sharegpt')
        return
    import pyarrow.parquet as pq
    paths = sorted(Path(source).glob('**/train-*-of-00006-*.parquet'))
    if len(paths) != 6:
        raise SystemExit(f'expected the six LMSYS-Chat-1M train shards under {source}, found {len(paths)}')
    index = 0
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=256,
                                                       columns=['conversation_id', 'conversation', 'redacted']):
            for row in batch.to_pylist():
                if row.get('redacted') is not True:
                    yield index, row.get('conversation_id'), final_user_prefix(row.get('conversation'), 'lmsys')
                index += 1


def main(a):
    models = dict(m.split('=', 1) for m in a.model)
    tokenizers = {name: AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=True)
                  for name, path in models.items()}
    longest = max(a.lengths)
    needed = 2 * a.world * max(a.batches)
    selected, seen, scanned = [], set(), 0
    for source_row, conversation_id, messages in source_rows(a.dataset, a.source):
        scanned += 1
        if not conversation_id or conversation_id in seen or not messages:
            continue
        ids = {}
        try:
            for name, tokenizer in tokenizers.items():
                extra = {'enable_thinking': False} if name.lower().startswith('qwen3') else {}
                ids[name] = tokenizer.apply_chat_template(messages, tokenize=True,
                                                          add_generation_prompt=True, **extra)
        except Exception:  # templates may reject malformed conversations
            continue
        if min(map(len, ids.values())) < longest:
            continue
        seen.add(conversation_id)
        selected.append((source_row, conversation_id, ids))
        if len(selected) == needed:
            break
    if len(selected) != needed:
        raise SystemExit(f'only {len(selected)} eligible conversations after {scanned} rows')
    out = Path(a.output)
    half = needed // 2
    cells = []
    for model in models:
        for length in a.lengths:
            for batch in a.batches:
                count = a.world * batch
                cell = f'{model}_{a.dataset}_R{a.world}_C{a.cache_percent}_B{batch}_L{length}_O{a.output_tokens}'
                files = {}
                for phase, chosen in (('target', selected[:count]), ('warmup', selected[half:half + count])):
                    requests = []
                    for index, (source_row, conversation_id, ids) in enumerate(chosen):
                        token_ids = ids[model][-length:]
                        requests.append(dict(request_id=source_row, source_row=source_row,
                                             conversation_id=conversation_id, global_index=index,
                                             origin_rank=index // batch, input_ids=token_ids,
                                             input_tokens=length))
                    path = out / model / 'manifests' / f'{cell}_{phase}.json'
                    write(path, dict(cell=cell, phase=phase, global_requests=count, input_tokens=length,
                                     output_tokens=a.output_tokens, requests=requests))
                    files[phase] = str(path.relative_to(out))
                    files[phase + '_sha256'] = digest(path)
                cells.append(dict(cell=cell, dataset=a.dataset, model=model, world=a.world, local_batch=batch,
                                  global_requests=count, input_tokens=length, output_tokens=a.output_tokens,
                                  cache_percent=a.cache_percent, **files))
    write(out / 'WORKLOADS.json', dict(dataset=a.dataset, models=list(models), world=a.world,
                                       scanned_source_rows=scanned, eligible_conversations=needed,
                                       selection=' '.join(__doc__.split('\n\n')[1].split()), cells=cells))
    print(f'{a.dataset}: {len(cells)} cells written to {out}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=('ShareGPT', 'LMSYS-Chat-1M'), required=True)
    p.add_argument('--source', required=True,
                   help='ShareGPT_V3_unfiltered_cleaned_split.json, or the LMSYS-Chat-1M dataset directory')
    p.add_argument('--model', action='append', required=True, help='NAME=PATH (repeat for every model)')
    p.add_argument('--output', required=True)
    p.add_argument('--world', type=int, default=4)
    p.add_argument('--batches', type=int, nargs='+', default=[16, 64], help='per-GPU batch sizes')
    p.add_argument('--lengths', type=int, nargs='+', default=[512, 1024], help='input lengths')
    p.add_argument('--output-tokens', type=int, default=64)
    p.add_argument('--cache-percent', type=int, default=30)
    main(p.parse_args())
