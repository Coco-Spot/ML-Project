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
PIPELINES="${PIPELINES:-direct_asr,diarization_asr,diarization_turn_asr}"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    PYTHON_BIN="python3"
  fi
fi

cd "${PROJECT_ROOT}"

mkdir -p "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/asr_benchmark/figures"

if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

OLD_CACHE_DIR="${PROJECT_ROOT}/outputs/caches"
if [[ "${MIGRATE_EXISTING_CACHE:-1}" == "1" && -d "${OLD_CACHE_DIR}" && ! -L "${OLD_CACHE_DIR}" ]]; then
  mkdir -p "$(dirname "${CACHE_ROOT}")"
  if [[ ! -e "${CACHE_ROOT}" ]]; then
    echo "Moving existing model cache to data disk: ${OLD_CACHE_DIR} -> ${CACHE_ROOT}"
    mv "${OLD_CACHE_DIR}" "${CACHE_ROOT}"
  else
    echo "Merging existing model cache into data disk: ${OLD_CACHE_DIR} -> ${CACHE_ROOT}"
    LEGACY_CACHE_ROOT="${CACHE_ROOT}_legacy_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "${CACHE_ROOT}" "${LEGACY_CACHE_ROOT}"
    shopt -s dotglob nullglob
    for cache_item in "${OLD_CACHE_DIR}"/*; do
      item_name="$(basename "${cache_item}")"
      if [[ -e "${CACHE_ROOT}/${item_name}" ]]; then
        mv "${cache_item}" "${LEGACY_CACHE_ROOT}/${item_name}"
      else
        mv "${cache_item}" "${CACHE_ROOT}/${item_name}"
      fi
    done
    shopt -u dotglob nullglob
    rmdir "${OLD_CACHE_DIR}"
    rmdir "${LEGACY_CACHE_ROOT}" 2>/dev/null || true
  fi
  ln -sfn "${CACHE_ROOT}" "${OLD_CACHE_DIR}"
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
  --output-prefix alimeeting
)
for model in "${MODEL_LIST[@]}"; do
  BUILD_ARGS+=(--asr-model "${model}")
done
IFS=',' read -r -a PIPELINE_LIST <<< "${PIPELINES}"
for pipeline in "${PIPELINE_LIST[@]}"; do
  BUILD_ARGS+=(--pipeline "${pipeline}")
done

if [[ "${SCAN_ONLY:-0}" == "1" ]]; then
  "${PYTHON_BIN}" server_experiments/alimeeting_trs/build_alimeeting_configs.py     --dataset-root "${DATASET_ROOT}"     --scan-only
  exit 0
fi

if ! "${PYTHON_BIN}" server_experiments/alimeeting_trs/build_alimeeting_configs.py "${BUILD_ARGS[@]}"; then
  echo ""
  echo "Config generation failed. Dataset diagnostics:"
  "${PYTHON_BIN}" server_experiments/alimeeting_trs/build_alimeeting_configs.py     --dataset-root "${DATASET_ROOT}"     --scan-only || true
  echo ""
  echo "Tip: set DATASET_ROOT to the AliMeeting directory that contains wav/TextGrid files."
  echo "Example: DATASET_ROOT=/root/autodl-tmp/moved/datasets/AliMeeting bash server_experiments/alimeeting_trs/run_alimeeting_trs.sh"
  exit 1
fi

RESULT_FILES=()
for index in "${!MODEL_LIST[@]}"; do
  model="${MODEL_LIST[$index]}"
  gpu="${GPU_LIST[$index]:-${GPU_LIST[0]:-0}}"
  slug="$(slug_model "${model}")"
  config="server_experiments/alimeeting_${slug}.json"
  output_dir="${OUTPUT_ROOT}/alimeeting_${slug}"
  log_path="${OUTPUT_ROOT}/logs/alimeeting_${slug}.log"

  echo "Running ${model} on GPU ${gpu}..."
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" -m overlap_asr_llm.cli run \
    --config "${config}" \
    --incremental \
    > "${log_path}" 2>&1

  echo "Evaluating ${model}..."
  CUDA_VISIBLE_DEVICES="${EVAL_GPU}" "${PYTHON_BIN}" -m overlap_asr_llm.cli evaluate \
    --config "${config}" \
    --results "${output_dir}/results.json" \
    --device "${BERT_DEVICE}" \
    --batch-size "${BERT_BATCH_SIZE}"

  RESULT_FILES+=("${output_dir}/readability_results.csv")
done

"${PYTHON_BIN}" server_experiments/alimeeting_trs/summarize_trs_selection.py \
  --results "${RESULT_FILES[@]}" \
  --output "${OUTPUT_ROOT}/asr_benchmark/alimeeting_trs_selection_report.md" \
  --combined-csv "${OUTPUT_ROOT}/asr_benchmark/readability_results_all.csv" \
  --summary-csv "${OUTPUT_ROOT}/asr_benchmark/model_pipeline_summary.csv" \
  --model-summary-csv "${OUTPUT_ROOT}/asr_benchmark/model_summary.csv"

"${PYTHON_BIN}" server_experiments/alimeeting_trs/plot_asr_benchmark.py \
  --model "" \
  --input "${OUTPUT_ROOT}/asr_benchmark/readability_results_all.csv" \
  --summary "${OUTPUT_ROOT}/asr_benchmark/model_pipeline_summary.csv" \
  --output-dir "${OUTPUT_ROOT}/asr_benchmark/figures"

echo "Done."
echo "- ${OUTPUT_ROOT}/asr_benchmark/readability_results_all.csv"
echo "- ${OUTPUT_ROOT}/asr_benchmark/model_pipeline_summary.csv"
echo "- ${OUTPUT_ROOT}/asr_benchmark/model_summary.csv"
echo "- ${OUTPUT_ROOT}/asr_benchmark/alimeeting_trs_selection_report.md"
echo "- ${OUTPUT_ROOT}/asr_benchmark/figures/"
