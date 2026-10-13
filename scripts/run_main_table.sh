#!/usr/bin/env bash
# Run every main-table cell (2 models x 2 datasets x 2 input lengths x 2 batch sizes)
# for FERRY-Random and FERRY. Finished cells (result.json present) are skipped.
#
# Required environment:
#   MODEL_DIR   directory holding Qwen3-30B-A3B-Instruct-2507/ and DeepSeek-V2-Lite-Chat/
#   WORK_DIR    output directory of scripts/prepare_workloads.py (holds ShareGPT/, LMSYS-Chat-1M/)
#   TOPO_DIR    output directory of the topology preprocessing (qwen3.json, deepseek.json)
# Optional:
#   GPUS (default 0,1,2,3), RESULTS (default results), REPEATS (default 3),
#   DATASETS, MODELS, LENGTHS, BATCHES, POLICIES to run a subset.
set -euo pipefail
cd "$(dirname "$0")/.."

: "${MODEL_DIR:?set MODEL_DIR}" "${WORK_DIR:?set WORK_DIR}" "${TOPO_DIR:?set TOPO_DIR}"
GPUS=${GPUS:-0,1,2,3}
RESULTS=${RESULTS:-results}
REPEATS=${REPEATS:-3}
DATASETS=${DATASETS:-"ShareGPT LMSYS-Chat-1M"}
MODELS=${MODELS:-"Qwen3 DeepSeekV2Lite"}
LENGTHS=${LENGTHS:-"512 1024"}
BATCHES=${BATCHES:-"16 64"}
POLICIES=${POLICIES:-"random ferry"}
NPROC=$(awk -F, '{print NF}' <<< "$GPUS")

declare -A MODEL_PATH=([Qwen3]="$MODEL_DIR/Qwen3-30B-A3B-Instruct-2507" [DeepSeekV2Lite]="$MODEL_DIR/DeepSeek-V2-Lite-Chat")
declare -A TOPOLOGY=([Qwen3]="$TOPO_DIR/qwen3.json" [DeepSeekV2Lite]="$TOPO_DIR/deepseek.json")

for dataset in $DATASETS; do
 for model in $MODELS; do
  for length in $LENGTHS; do
   for batch in $BATCHES; do
    cell="${model}_${dataset}_R${NPROC}_C30_B${batch}_L${length}_O64"
    for policy in $POLICIES; do
     out="$RESULTS/$cell/$policy"
     [ -f "$out/result.json" ] && { echo "skip $cell $policy"; continue; }
     mkdir -p "$out"
     echo "$(date +%T) run $cell $policy"
     CUDA_VISIBLE_DEVICES=$GPUS python -m torch.distributed.run --standalone --nproc_per_node "$NPROC" \
       scripts/run_cell.py --model-path "${MODEL_PATH[$model]}" --workloads "$WORK_DIR/$dataset/WORKLOADS.json" \
       --cell "$cell" --policy "$policy" --topology "${TOPOLOGY[$model]}" --repeats "$REPEATS" \
       --output "$out" > "$out/run.log" 2>&1 || { echo "FAILED $cell $policy (see $out/run.log)"; exit 1; }
    done
   done
  done
 done
done
python scripts/summarize.py --results "$RESULTS" --output "$RESULTS/main_table.md"
