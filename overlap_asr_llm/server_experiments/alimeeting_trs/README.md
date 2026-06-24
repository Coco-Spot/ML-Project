# AliMeeting/M2MeT ASR Benchmark Server Experiment

This folder is a standalone server experiment add-on. It does not change the
existing project source code or existing configs.

## Goal

Benchmark the raw recognition ability of three Chinese meeting-speech ASR
backends under the same AliMeeting/M2MeT samples and pipelines:

- `whisper:large-v3`
- `faster-whisper:large-v3`
- `funasr`

The benchmark also keeps speaker-aware metrics so the results can support a
paper discussion about overlapping multi-speaker meetings.

## Pipelines

Each ASR model is evaluated with the same non-LLM pipelines:

- `direct_asr`
- `diarization_asr`
- `diarization_turn_asr`
- `separation_asr`

## Server Paths

Expected defaults:

```bash
PROJECT_ROOT=/root/ML-Project/overlap_asr_llm
DATASET_ROOT=/root/autodl-tmp/moved/datasets/AliMeeting
```

## One-command Run

From the server:

```bash
cd /root/ML-Project/overlap_asr_llm
bash server_experiments/alimeeting_trs/run_alimeeting_trs.sh
```

The default model set is:

```bash
ASR_MODELS=whisper:large-v3,faster-whisper:large-v3,funasr
```

Useful controls:

```bash
MAX_HOURS=1 \
ASR_MODELS=whisper:large-v3,faster-whisper:large-v3,funasr \
ASR_GPUS=0,0,0 \
EVAL_GPU=0 \
BERT_DEVICE=cuda \
bash server_experiments/alimeeting_trs/run_alimeeting_trs.sh
```

`MAX_HOURS=2` is the default validation subset size. The runner executes models
sequentially by default, which avoids GPU contention and makes runtime
comparisons cleaner.

## Required Environment

`pyannote` needs a Hugging Face token that has accepted the pyannote model terms.
Put it in `/root/ML-Project/overlap_asr_llm/.env` or export it before running:

```bash
export HF_TOKEN=...
```

Install project dependencies first if the server environment is fresh:

```bash
cd /root/ML-Project/overlap_asr_llm
pip install -r requirements.txt
pip install -e .
```

For figure generation, `matplotlib` is required.

## Generated Configs

The config builder writes one config per model, for example:

```text
server_experiments/alimeeting_trs_whisper_large_v3.json
server_experiments/alimeeting_trs_faster_whisper_large_v3.json
server_experiments/alimeeting_trs_funasr.json
```

You can generate configs only:

```bash
python server_experiments/alimeeting_trs/build_alimeeting_configs.py \
  --dataset-root /root/autodl-tmp/moved/datasets/AliMeeting \
  --asr-model whisper:large-v3 \
  --asr-model faster-whisper:large-v3 \
  --asr-model funasr
```

## Generated Results

The full run writes:

```text
sever_outputs/alimeeting_whisper_large_v3/
sever_outputs/alimeeting_faster_whisper_large_v3/
sever_outputs/alimeeting_funasr/
sever_outputs/asr_benchmark/readability_results_all.csv
sever_outputs/asr_benchmark/model_pipeline_summary.csv
sever_outputs/asr_benchmark/model_summary.csv
sever_outputs/asr_benchmark/alimeeting_trs_selection_report.md
sever_outputs/asr_benchmark/figures/
```

Paper-oriented figures:

```text
sever_outputs/asr_benchmark/figures/cer_by_model_overlap.png
sever_outputs/asr_benchmark/figures/trs_text_by_model.png
sever_outputs/asr_benchmark/figures/trs_speaker_heatmap.png
sever_outputs/asr_benchmark/figures/runtime_by_model.png
```

Recommended paper usage:

- Use `cer_by_model_overlap.png` to show recognition robustness as overlap increases.
- Use `trs_text_by_model.png` for overall transcript usefulness.
- Use `trs_speaker_heatmap.png` to discuss model and pipeline interaction for speaker-attributed transcripts.
- Use `runtime_by_model.png` as a practical efficiency comparison.

## Pipeline Figures

Use this after running any ASR model across the experiment pipelines. The plotting
script reads a row-level `readability_results.csv` file and writes pipeline
comparison figures for the selected model.

For the default generated faster-whisper directory:

```bash
python server_experiments/alimeeting_trs/plot_asr_benchmark.py
```

To switch models, set the model name and the matching output directory:

```bash
MODEL="faster-whisper:large-v3"
RUN_DIR="sever_outputs/alimeeting_faster_whisper_large_v3"

python server_experiments/alimeeting_trs/plot_asr_benchmark.py \
  --model "${MODEL}" \
  --input "${RUN_DIR}/readability_results.csv" \
  --output-dir "${RUN_DIR}/figures"
```

Examples:

```bash
MODEL="whisper:large-v3"
RUN_DIR="sever_outputs/alimeeting_whisper_large_v3"

MODEL="funasr"
RUN_DIR="sever_outputs/alimeeting_funasr"
```

If the input CSV contains only one model, the `--model` value can match that
model. If the input CSV contains multiple ASR models, `--model` selects one; pass
`--model ""` to plot all rows together.

Generated pipeline figures:

```text
${RUN_DIR}/figures/cer_by_pipeline_overlap.png
${RUN_DIR}/figures/trs_text_by_pipeline.png
${RUN_DIR}/figures/trs_speaker_by_pipeline.png
${RUN_DIR}/figures/runtime_by_pipeline.png
```

## Notes

The config builder expects Praat TextGrid annotations somewhere under the dataset
root with the same file stem as each `.wav`. If no annotated samples are found,
inspect the dataset layout and pass the correct root directory. For a pure smoke
run without TRS references:

```bash
python server_experiments/alimeeting_trs/build_alimeeting_configs.py \
  --dataset-root /root/autodl-tmp/moved/datasets/AliMeeting \
  --allow-unannotated
```

## Direct ASR benchmark only

Use this when you only want the ASR model comparison without diarization or
separation outputs:

```bash
cd /root/ML-Project/overlap_asr_llm
ASR_MODELS=whisper:large-v3,faster-whisper:large-v3,funasr   bash server_experiments/alimeeting_trs/run_direct_asr_benchmark.sh
```

This writes one benchmark config that you can edit before rerunning:

```text
server_experiments/direct_asr_benchmark.json
```

The generated config compares all requested models in `asr_models` and contains only:

```json
"pipelines": ["direct_asr"]
```

Paper-facing summary outputs go under:

```text
sever_outputs/direct_asr_benchmark/results.csv
sever_outputs/direct_asr_benchmark/readability_results.csv
sever_outputs/direct_asr_benchmark/readability_results_all.csv
sever_outputs/direct_asr_benchmark/model_pipeline_summary.csv
sever_outputs/direct_asr_benchmark/model_summary.csv
sever_outputs/direct_asr_benchmark/direct_asr_selection_report.md
sever_outputs/direct_asr_benchmark/figures/
```

## Prune outputs for paper writing

Preview what would be removed:

```bash
python server_experiments/alimeeting_trs/prune_outputs_for_paper.py
```

Actually remove non-paper artifacts such as separated audio and logs:

```bash
python server_experiments/alimeeting_trs/prune_outputs_for_paper.py --apply
```

Caches are preserved by default. Add `--include-caches` only when you also want
to delete model caches.

