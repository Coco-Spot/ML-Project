#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/ML-Project/overlap_asr_llm}"
DATASET_ROOT="${DATASET_ROOT:-/root/autodl-tmp/moved/datasets/AliMeeting}"
MAX_HOURS="${MAX_HOURS:-2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-sever_outputs}"
CACHE_ROOT="${CACHE_ROOT:-/root/autodl-tmp/overlap_asr_llm_cache}"
ASR_MODELS="${ASR_MODELS:-funasr}"
ASR_GPUS="${ASR_GPUS:-0}"
BERT_DEVICE="${BERT_DEVICE:-cuda}"
BERT_BATCH_SIZE="${BERT_BATCH_SIZE:-8}"
EVAL_GPU="${EVAL_GPU:-0}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-direct_asr_benchmark}"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    PYTHON_BIN="python3"
  fi
fi

cd "${PROJECT_ROOT}"

mkdir -p "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/direct_asr_benchmark/figures"

if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

export OVERLAP_ASR_LLM_ENABLE_TF32="${OVERLAP_ASR_LLM_ENABLE_TF32:-1}"
export OVERLAP_ASR_LLM_CACHE_DIR="${OVERLAP_ASR_LLM_CACHE_DIR:-${CACHE_ROOT}}"
export HF_HOME="${HF_HOME:-${CACHE_ROOT}/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${CACHE_ROOT}/huggingface/hub}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-${CACHE_ROOT}/modelscope}"

IFS=',' read -r -a MODEL_LIST <<< "${ASR_MODELS}"
IFS=',' read -r -a GPU_LIST <<< "${ASR_GPUS}"

slug_model() {
  "${PYTHON_BIN}" -c 'import re,sys; s=sys.argv[1].lower().replace(":","_").replace("/","_"); print(re.sub(r"_+","_",re.sub(r"[^a-z0-9_]+","_",s)).strip("_"))' "$1"
}

BUILD_ARGS=(
  --dataset-root "${DATASET_ROOT}"
  --max-hours "${MAX_HOURS}"
  --output-dir server_experiments
  --experiment-output-root "${OUTPUT_ROOT}"
  --single-config "${OUTPUT_PREFIX}"
  --single-config-output "${OUTPUT_PREFIX}"
  --experiment-label "AliMeeting/M2MeT direct ASR benchmark"
  --pipeline direct_asr
)
for model in "${MODEL_LIST[@]}"; do
  BUILD_ARGS+=(--asr-model "${model}")
done

if [[ "${SCAN_ONLY:-0}" == "1" ]]; then
  "${PYTHON_BIN}" server_experiments/alimeeting_trs/build_alimeeting_configs.py --dataset-root "${DATASET_ROOT}" --scan-only
  exit 0
fi

"${PYTHON_BIN}" server_experiments/alimeeting_trs/build_alimeeting_configs.py "${BUILD_ARGS[@]}"

config="server_experiments/${OUTPUT_PREFIX}.json"
output_dir="${OUTPUT_ROOT}/${OUTPUT_PREFIX}"
log_path="${OUTPUT_ROOT}/logs/${OUTPUT_PREFIX}.log"

echo "Running direct_asr benchmark for: ${ASR_MODELS}"
CUDA_VISIBLE_DEVICES="${ASR_GPUS%%,*}" "${PYTHON_BIN}" -m overlap_asr_llm.cli run   --config "${config}"   --incremental   > "${log_path}" 2>&1

echo "Evaluating direct_asr benchmark..."
CUDA_VISIBLE_DEVICES="${EVAL_GPU}" "${PYTHON_BIN}" -m overlap_asr_llm.cli evaluate   --config "${config}"   --results "${output_dir}/results.json"   --device "${BERT_DEVICE}"   --batch-size "${BERT_BATCH_SIZE}"

"${PYTHON_BIN}" server_experiments/alimeeting_trs/summarize_trs_selection.py   --results "${output_dir}/readability_results.csv"   --output "${output_dir}/direct_asr_selection_report.md"   --combined-csv "${output_dir}/readability_results_all.csv"   --summary-csv "${output_dir}/model_pipeline_summary.csv"   --model-summary-csv "${output_dir}/model_summary.csv"

"${PYTHON_BIN}" server_experiments/alimeeting_trs/plot_asr_benchmark.py   --model ""   --input "${output_dir}/readability_results_all.csv"   --summary "${output_dir}/model_pipeline_summary.csv"   --output-dir "${output_dir}/figures"

echo "Done."
echo "- ${OUTPUT_ROOT}/direct_asr_benchmark/readability_results_all.csv"
echo "- ${OUTPUT_ROOT}/direct_asr_benchmark/model_pipeline_summary.csv"
echo "- ${OUTPUT_ROOT}/direct_asr_benchmark/model_summary.csv"
echo "- ${OUTPUT_ROOT}/direct_asr_benchmark/direct_asr_selection_report.md"
echo "- ${OUTPUT_ROOT}/direct_asr_benchmark/figures/"
