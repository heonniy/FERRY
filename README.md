# FERRY: Dual-Strategy Multi-GPU MoE Offloading

This repository reproduces the FERRY and FERRY-Random rows of the main table (Table 2):
Qwen3-30B-A3B-Instruct-2507 and DeepSeek-V2-Lite-Chat on 4 GPUs with a 30% expert-cache
budget, ShareGPT and LMSYS-Chat-1M, input lengths 512 / 1024, per-GPU batch sizes 16 / 64,
64 output tokens.

| Method | Step 2: how many missing experts each GPU fetches | Step 3: which GPU gets which expert | Eviction |
|---|---|---|---|
| FERRY | topology-aware quota from profiled H2D bandwidth (Eq. 6, Alg. 1 lines 1-5) | load-balanced, locality-aware assignment (Eq. 9-10, Alg. 1 lines 6-14) | selection frequency over the last W = 128 routed tokens, ties by LRU |
| FERRY-Random | none: each missing expert goes to a uniformly random GPU | random | same as FERRY |

Both methods share the same runtime (data-parallel attention, expert-parallel routed
experts, per-GPU expert caches, BF16) and apply their policy at every MoE layer, in
prefill and decode.

## Code map

| Paper | Code |
|---|---|
| §5.1 global demand aggregation, App. A.1 metadata | `ferry/metadata.py` (`RouteGather`), `ferry/policy.py` (`_demand`) |
| §5.2 topology-aware allocation, bandwidth profiling | `scripts/profile_h2d.py`, `scripts/build_quota_lut.py`, `ferry/quota.py` |
| §5.3 load-balanced assignment | `ferry/policy.py` (`load_balanced_assignment`) |
| §5.4 routing-aware eviction | `ferry/metadata.py` (`GateHistory`), `ferry/policy.py` (`_choose_slot`) |
| EP dispatch / combine | `ferry/layout.py`, `ferry/transport.py` |
| expert fetch and compute | `ferry/h2d.py`, `ferry/csrc/expert_ops.{cpp,cu}`, `ferry/runtime.py` |
| model loading and MoE hooks | `ferry/model.py` |
| workloads, measurement, table | `scripts/prepare_workloads.py`, `scripts/run_cell.py`, `scripts/run_main_table.sh`, `scripts/summarize.py` |

## Requirements

- Linux, 4 NVIDIA GPUs with 80 GB each (tested on H100 SXM; NVLink or PCIe both work), CUDA toolkit
  12.x with `nvcc` (the expert kernel is compiled on the first run).
- Host memory: every GPU process keeps a pinned copy of all routed experts
  (Qwen3: 54 GiB per GPU; DeepSeek-V2-Lite: 27 GiB per GPU), i.e. at least ~256 GB free RAM for Qwen3 on 4 GPUs.
- Python 3.12 and the packages in `requirements.txt`:

```bash
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

## 1. Download models and datasets

Run every command from the repository root, in one shell, so that the exported variables
(`MODEL_DIR`, `DATA_DIR`, `WORK_DIR`, `GPUS`, `TOPO_DIR`) persist across steps.


```bash
export MODEL_DIR=/path/to/models DATA_DIR=/path/to/datasets
huggingface-cli download Qwen/Qwen3-30B-A3B-Instruct-2507 --local-dir $MODEL_DIR/Qwen3-30B-A3B-Instruct-2507
huggingface-cli download deepseek-ai/DeepSeek-V2-Lite-Chat --local-dir $MODEL_DIR/DeepSeek-V2-Lite-Chat
huggingface-cli download anon8231489123/ShareGPT_Vicuna_unfiltered ShareGPT_V3_unfiltered_cleaned_split.json \
    --repo-type dataset --local-dir $DATA_DIR/ShareGPT
# LMSYS-Chat-1M is gated: accept its license on the Hugging Face page and `huggingface-cli login` first.
huggingface-cli download lmsys/lmsys-chat-1m --repo-type dataset --local-dir $DATA_DIR/lmsys-chat-1m
```

## 2. Build the request manifests

The selection is deterministic (no sampling): the first 512 distinct conversations whose rendered
prompt (history up to the last non-empty user turn) has at least 1,024 tokens under both chat
templates. Targets are the first `4 x batch` conversations, warmup requests are taken from the
second 256; each request keeps its last 512 or 1,024 tokens.

```bash
export WORK_DIR=$PWD/workloads
MODEL_ARGS="--model Qwen3=$MODEL_DIR/Qwen3-30B-A3B-Instruct-2507 --model DeepSeekV2Lite=$MODEL_DIR/DeepSeek-V2-Lite-Chat"
python scripts/prepare_workloads.py --dataset ShareGPT \
    --source $DATA_DIR/ShareGPT/ShareGPT_V3_unfiltered_cleaned_split.json $MODEL_ARGS --output $WORK_DIR/ShareGPT
python scripts/prepare_workloads.py --dataset LMSYS-Chat-1M \
    --source $DATA_DIR/lmsys-chat-1m $MODEL_ARGS --output $WORK_DIR/LMSYS-Chat-1M
```

## 3. Profile H2D bandwidth and build the Step 2 lookup table

Pick the 4 GPUs once (integer indices) and use the same `CUDA_VISIBLE_DEVICES` (same order) in
every later step; `run_cell.py` refuses a topology file profiled on other GPUs.
Run this on an otherwise idle machine. For every subset of the GPUs it measures concurrent
expert-sized pinned-host-to-GPU copies, then turns the measured bandwidth domains into a
quota table `quota_lut[m]` (m = number of missing experts in a layer) using the greedy of
Algorithm 1. The runtime reads the table at every layer and re-runs the greedy only if a
row would exceed some GPU's free/evictable cache slots.

```bash
export GPUS=0,1,2,3 TOPO_DIR=$PWD/topology
for m in qwen3:Qwen3-30B-A3B-Instruct-2507 deepseek:DeepSeek-V2-Lite-Chat; do
  CUDA_VISIBLE_DEVICES=$GPUS python scripts/profile_h2d.py --model-path $MODEL_DIR/${m#*:} \
      --output $TOPO_DIR/${m%%:*}_profile.json
  python scripts/build_quota_lut.py --profile $TOPO_DIR/${m%%:*}_profile.json \
      --model-path $MODEL_DIR/${m#*:} --output $TOPO_DIR/${m%%:*}.json
done
```

By default every profiled GPU subset is a bandwidth domain. To restrict the shared domains
to known PCIe switches or NUMA nodes, pass e.g. `--groups 0,1:2,3` to `build_quota_lut.py`.
When all GPUs have the same bandwidth and nothing is shared, the table is an even split.

## 4. Smoke test (about 3 minutes)

```bash
CUDA_VISIBLE_DEVICES=$GPUS python -m torch.distributed.run --standalone --nproc_per_node 4 scripts/run_cell.py \
    --model-path $MODEL_DIR/DeepSeek-V2-Lite-Chat --workloads $WORK_DIR/ShareGPT/WORKLOADS.json \
    --cell DeepSeekV2Lite_ShareGPT_R4_C30_B16_L512_O64 --policy ferry --topology $TOPO_DIR/deepseek.json \
    --output results/smoke --smoke
```

## 5. Run the main table

```bash
scripts/run_main_table.sh          # uses MODEL_DIR, WORK_DIR, TOPO_DIR, GPUS
```

This runs 32 jobs (16 cells x 2 methods). Each job loads the model once, runs one warmup
generation on the warmup requests and one unreported generation on the target requests
(the first pass over new tensor shapes is slow, mostly in TTFT), then `REPEATS=3` measured
generations on the target requests. Every generation starts from an empty expert cache. Completed jobs are skipped, so the script
can be resumed. Subsets can be selected with `MODELS`, `DATASETS`, `LENGTHS`, `BATCHES`,
`POLICIES`, e.g. `MODELS=DeepSeekV2Lite BATCHES=16 scripts/run_main_table.sh`.

The script writes `results/main_table.md` at the end. To regenerate it (e.g. after a subset run):

```bash
python scripts/summarize.py --results results --output results/main_table.md
```

## Metrics

All ranks start from a common release time. With `first` the time the last rank has its
first token and `end` the time the last rank finishes:

- TTFT = first - release;
- TPOT = (end - first) / 63;
- TPS = global requests x 64 / (end - release).

Values are reported as mean ± sample standard deviation over the measured repeats.
Per-repeat and per-rank records (fetch counts per GPU, H2D copies, communication bytes,
peak memory) are in `results/<cell>/<method>/`.

## Execution details

- Serial execution: on every GPU the routed-expert GEMMs of a layer start only after all of
  that GPU's H2D fetches for the layer have completed. Fetches overlap only with the dispatch
  all-to-all.
- Dense weights (attention, router, shared experts) are GPU-resident; routed experts are
  fetched on demand from pinned host memory into fixed-size cache slots.
- Cache budget: `floor(0.30 x layers x experts)` expert slots, split evenly over the GPUs.
- Step 3 option `--balance-tol-bps` (default 200 = 2%): ranks whose projected peak load is within this
  tolerance of the minimum are treated as equally balanced, and the expert goes to the one
  holding most of its tokens. `--balance-tol-bps 0` is the strict argmin rule of Eq. 9-10.
- `--cpu-affinity auto` (default) pins each process to a disjoint share of the host CPUs
  chosen by physical GPU index; this noticeably reduces timing noise.
- Eviction score: `--eviction-score frequency` (default) is the top-k selection frequency of each
  expert over the W most recent routed tokens (Eq. 13), taken evenly from all GPUs; ties are broken
  by least-recent use. `--eviction-score probability` uses the mean router probability instead
  (Eq. 12) and additionally all-gathers router probabilities.
- FERRY-Random draws ranks from a per-layer seed (`--salt`), identical on every GPU.

## Baselines

DeepSpeed-Inference, MoE-Infinity, and llama.cpp were run from their official releases with
the configurations described in Appendix D.2 and the same request manifests.
