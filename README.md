# Overlapping Speech ASR + LLM Project

This repository contains a reproducible experiment suite for overlapping
Mandarin speech recognition. It compares direct ASR, speaker diarization,
speech separation, and constrained LLM/RAG transcript refinement on a controlled
two-speaker debate dataset.

## Research Question

When two speakers overlap, which pipeline produces the most useful transcript?

The current experiment compares five paths:

1. `direct_asr`: transcribe mixed audio directly.
2. `diarization_asr`: transcribe full mixed audio, then align speaker labels.
3. `diarization_turn_asr`: diarize first, then transcribe each speaker turn.
4. `separation_asr`: separate speakers first, then transcribe each stream.
5. `llm_rag_refine`: use project context to clean ASR outputs without inventing
   missing words.

## Current Status

The main server run is complete for five Mandarin `sample2` conditions:

- no overlap
- light overlap
- medium overlap
- heavy overlap
- opposite-order overlap

The final comparison config is
`overlap_asr_llm/configs/all_pipelines.json`. It uses:

- ASR: `whisper:large-v3`
- Diarization: `pyannote/speaker-diarization-community-1`
- Separation: `clearvoice:MossFormer2_SS_16K`
- LLM/RAG: OpenAI-compatible `api` provider

Headline results from `outputs/all_pipelines/run_summary.md` and
`outputs/all_pipelines/readability_summary.md`:

- Best average CER/WER and TRS text: `diarization_asr`, avg CER `0.1532`,
  avg WER `0.1563`, avg TRS text `85.8221`.
- Fastest pipeline: `direct_asr`, avg runtime `6.72s`.
- Best no-overlap row: `diarization_turn_asr`, CER `0.0094`.
- Best opposite-overlap row: `separation_asr`, CER `0.0632`, WER `0.0769`,
  TRS text `89.0998`.
- LLM/RAG improves readability on light, medium, and heavy overlap, but should
  be reported as constrained cleanup rather than reliable recovery of missing
  speech.

## Quick Start

For team members, the Chinese quick start is:
`overlap_asr_llm/docs/QUICK_START_FOR_TEAM_CN.md`

From the repository root, the common workflows now have short commands:

```bash
make app                    # open the frontend
make mock                   # run the lightweight mock pipeline
make smoke                  # run the smoke test
make exp                    # run configs/all_pipelines.json
make exp CONFIG=direct_asr  # run one experiment config
make eval                   # evaluate outputs/all_pipelines/results.json
make test                   # run unit tests
make package                # build a safe submission zip
```

Run the local mock pipeline first. It uses only lightweight mock providers and
is meant to verify that the repository works:

```bash
cd overlap_asr_llm
python3 -m overlap_asr_llm.cli run --config configs/mock.json --mock
```

The command writes:

- `outputs/mock/results.json`
- `outputs/mock/results.csv`
- `outputs/mock/run_summary.md`

You can also run the stdlib smoke test:

```bash
python3 scripts/smoke_test.py
```

## Installation

The mock pipeline only needs Python 3.10+ and the source tree. For editable
development:

```bash
cd overlap_asr_llm
pip install -e .
```

For real-model experiments, install the full optional stack:

```bash
pip install -r requirements.txt
pip install -e .
```

Real ASR, diarization, separation, and readability evaluation may require GPU
memory, large model downloads, and API credentials.

## Docker

The Dockerfile lives in `overlap_asr_llm/Dockerfile` and is meant to give the
instructor a reproducible way to run tests, mock workflows, or the frontend
without manually configuring Python dependencies.

Build the full image from the repository root:

```bash
docker build -t overlap-asr-llm ./overlap_asr_llm
```

Run the default unit-test check:

```bash
docker run --rm overlap-asr-llm
```

Run the lightweight mock pipeline:

```bash
docker run --rm overlap-asr-llm \
  python -m overlap_asr_llm.cli run --config configs/mock.json --mock
```

Run the frontend on `http://127.0.0.1:7861`:

```bash
docker run --rm -p 7861:7861 overlap-asr-llm \
  python scripts/launch_speaker_app.py --host 0.0.0.0 --port 7861
```

For real-model experiments, pass credentials as environment variables or mount
a local `.env` file. Do not bake secrets or downloaded model caches into the
image.

```bash
docker run --rm --env-file overlap_asr_llm/.env overlap-asr-llm \
  python -m overlap_asr_llm.cli run --config configs/all_pipelines.json --incremental
```

## Environment Setup

Before running real-model experiments or the frontend, create
`overlap_asr_llm/.env` and put your own credentials there. This file is local
only: do not commit it and do not include it in submission packages.

Minimum example:

```bash
# Required for pyannote diarization.
# Get this from the original Hugging Face website after accepting the model terms.
HF_TOKEN=hf_your_token_here

# Required for OpenAI-compatible LLM/RAG cleanup.
# DeepSeek is recommended for this project; use your own API key.
OPENAI_API_KEY=sk_your_deepseek_api_key_here
OPENAI_BASE_URL=https://api.deepseek.com
OPENAI_MODEL=deepseek-v4-pro
```

Important pyannote note: `pyannote/speaker-diarization-community-1` is a gated
Hugging Face model. You must log in on the original Hugging Face model page,
accept the user conditions, create a Hugging Face access token, and then place
that token in `.env` as `HF_TOKEN` or `HUGGINGFACE_TOKEN`. Do not rely on mirror
sites for this model, because mirrors cannot grant the required model access
permission to your Hugging Face account.

Important LLM note: the `api` LLM provider reads `OPENAI_API_KEY`,
`OPENAI_BASE_URL`, and `OPENAI_MODEL` from `.env`. For DeepSeek V4 Pro, use the
OpenAI-compatible endpoint above and set `OPENAI_MODEL=deepseek-v4-pro`.

## Running Experiments

Short version from the repository root:

```bash
make exp
make exp CONFIG=direct_asr
make exp CONFIG=diarization_asr
make exp CONFIG=separation_asr
make exp CONFIG=speaker_llm_pipeline
```

Run the current final comparison:

```bash
cd overlap_asr_llm
python3 -m overlap_asr_llm.cli run --config configs/all_pipelines.json --incremental
```

Run a single pipeline config:

```bash
python3 -m overlap_asr_llm.cli run --config configs/direct_asr.json --incremental
python3 -m overlap_asr_llm.cli run --config configs/diarization_asr.json --incremental
python3 -m overlap_asr_llm.cli run --config configs/separation_asr.json --incremental
python3 -m overlap_asr_llm.cli run --config configs/speaker_llm_pipeline.json --incremental
```

Run post-hoc readability evaluation after a result JSON exists:

```bash
make eval
```

The long equivalent is:

```bash
python3 -m overlap_asr_llm.cli evaluate \
  --config configs/all_pipelines.json \
  --results outputs/all_pipelines/results.json \
  --device auto \
  --batch-size 1
```

The evaluator writes `readability_results.csv`,
`readability_results.json`, and `readability_summary.md` beside the input
`results.json`.

## Providers

Supported provider values:

- ASR: `mock`, `whisper`, `whisper:<model-name>`,
  `faster-whisper`, `faster-whisper:<model-name>`, `funasr`
- Diarization: `mock`, `pyannote`, `pyannote:<huggingface-model-id>`,
  `speechbrain`, `speechbrain:<huggingface-model-id>`
- Separation: `mock`, `sepformer`, `sepformer:<huggingface-model-id>`,
  `clearvoice`, `clearvoice:<model-name>`
- LLM/RAG refinement: `mock`, `api`, `api:<model-name>`

The `--mock` flag deliberately forces all providers to mock mode. Real LLM/RAG
refinement uses the OpenAI-compatible `api` provider and requires
`OPENAI_API_KEY`. Optional `OPENAI_BASE_URL` and `OPENAI_MODEL` environment
variables are also supported. For the current recommended DeepSeek setup, put
`OPENAI_BASE_URL=https://api.deepseek.com` and
`OPENAI_MODEL=deepseek-v4-pro` in `overlap_asr_llm/.env`.

## Outputs Kept In GitHub

The repository keeps selected outputs needed for review and reproduction:

- `overlap_asr_llm/outputs/all_pipelines/results.csv`
- `overlap_asr_llm/outputs/all_pipelines/results.json`
- `overlap_asr_llm/outputs/all_pipelines/run_summary.md`
- `overlap_asr_llm/outputs/all_pipelines/readability_summary.md`
- `overlap_asr_llm/outputs/all_pipelines/readability_results.csv`
- `overlap_asr_llm/outputs/all_pipelines/readability_results.json`
- `overlap_asr_llm/outputs/all_pipelines/diarization_segments.*`
- `overlap_asr_llm/outputs/all_pipelines/separation_segments.*`
- `overlap_asr_llm/outputs/all_pipelines/separated_audio/`
- each single-pipeline output directory under `outputs/direct_asr/`,
  `outputs/diarization_asr/`, `outputs/separation_asr/`, and
  `outputs/speaker_llm_pipeline/`


## Repository Layout

```text
Project.md                                      Original course/project prompt
overlap_asr_llm/configs/                        Experiment configurations
overlap_asr_llm/data/samples/                   Mock/demo audio samples
overlap_asr_llm/data/samples2/                  Current five-condition Mandarin set
overlap_asr_llm/docs/EXPERIMENT_DESIGN.md       Experiment design and conclusions
overlap_asr_llm/docs/QUALITATIVE_FINDINGS.md    Manual failure-case notes
overlap_asr_llm/docs/TRUE_READABILITY_SCORE.md  Readability/TRS definition
overlap_asr_llm/docs/CORE_CODE_LINE_BY_LINE_CN.md
                                                Chinese line-by-line code guide
overlap_asr_llm/src/overlap_asr_llm/            Core package
overlap_asr_llm/scripts/                        Utility scripts
overlap_asr_llm/tests/                          Unit tests
overlap_asr_llm/outputs/                        Selected experiment outputs
overlap_asr_llm/CONTRIBUTIONS.md                Team contribution record
overlap_asr_llm/REPOSITORY.md                   Repository/submission notes
```

Core source files:

- `config.py`: config loading and config inheritance.
- `providers.py`: ASR, diarization, separation, and LLM provider wrappers.
- `pipelines.py`: experiment orchestration.
- `metrics.py`: CER/WER and speaker-block scoring.
- `io.py`: JSON, CSV, Markdown, HTML, and segment writers.
- `readability.py`: BERTScore/TRS readability evaluation.

## Metrics

The runner reports:

- CER and WER
- runtime
- speaker labels and segment exports
- `flat_cer` / `flat_wer`
- `timeline_cer` / `timeline_wer`
- `speaker_block_cer` / `speaker_block_wer`
- primary `cer` / `wer` selected by `score_basis`

Metric variants:

| Field | Meaning |
| --- | --- |
| `flat_cer` / `flat_wer` | Compare the flattened transcript text against the flattened reference. Speaker labels and timing are not the focus. |
| `timeline_cer` / `timeline_wer` | Sort output segments by timestamp, concatenate their text, then compare against the reference timeline. |
| `speaker_block_cer` / `speaker_block_wer` | Group text by speaker first, try the best mapping between predicted speaker labels and reference speakers, then compare speaker blocks. This is useful because diarization labels can be swapped, e.g. predicted `SPEAKER_00` may correspond to reference `speaker_2`. |
| `cer` / `wer` | The primary score selected for that row by `score_basis`; it may use timeline or speaker-block scoring depending on the pipeline output. |

Lower CER/WER values are better. `speaker_block` means "compare by speaker's
whole content block", not "compare by chronological turn". It rewards systems
that keep each speaker's words together even if the arbitrary speaker label
names are reversed.

The readability evaluator adds:

- BERTScore precision, recall, F1, and F2
- text-only TRS
- speaker-aware TRS where speaker segments are available
- overlap ratio reporting
- high-overlap review notes

TRS, or True Readability Score, is an exploratory reporting formula for this
project rather than a published standard metric. The motivation is that CER/WER
alone can miss whether a transcript is actually useful under overlap: a system
may have decent edit distance while dropping a speaker's clause, mixing speaker
turns, or adding unsupported cleanup. The current text-only version combines
character accuracy with recall-weighted semantic coverage:

```math
\mathrm{TRS}_{text}
=
100
\times
\sqrt{
\left(1-\min(\mathrm{CER},1)\right)
\times
F_{2,\mathrm{BERT}}
}
```

For speaker-attributed outputs, the speaker-aware version adds speaker-block
accuracy:

```math
\mathrm{TRS}_{speaker}
=
100
\times
\sqrt[3]{
\left(1-\min(\mathrm{CER},1)\right)
\times
F_{2,\mathrm{BERT}}
\times
\left(1-\min(\mathrm{SpeakerBlockCER},1)\right)
}
```

This geometric-mean design keeps the formula simple and makes one weak
dimension lower the final score, which matches the project goal: a transcript
should be accurate, semantically faithful, and speaker-readable at the same
time. The full motivation, component definitions, and interpretation notes are
in `overlap_asr_llm/docs/TRUE_READABILITY_SCORE.md`.

## Testing

Run the unit tests with:

```bash
make test
```

If `pytest` is installed:

```bash
make pytest
```

## Submission Notes

Before final submission:

- Fill in final team contribution evidence in `overlap_asr_llm/CONTRIBUTIONS.md`.
- Confirm `overlap_asr_llm/.env` exists locally for real-model runs, but do not
  commit it.
- Build the safe submission archive with `make package`. The packaging script
  excludes `.env`, virtual environments, model caches, temporary frontend
  outputs, and temporary experiment outputs.
- Keep only intended code, configs, docs, sample audio, and selected outputs in
  GitHub.
