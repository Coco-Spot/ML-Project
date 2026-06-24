#!/usr/bin/env python3
"""Organize AliMeeting benchmark outputs by benchmark view and ASR model."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import shutil
from statistics import mean


MODEL_SPECS = [
    ("funasr", "alimeeting_funasr", ("alimeeting_trs_funasr",)),
    (
        "whisper:large-v3",
        "alimeeting_whisper_large_v3",
        ("alimeeting_trs_whisper_large_v3",),
    ),
    (
        "faster-whisper:large-v3",
        "alimeeting_faster_whisper_large_v3",
        ("alimeeting_trs_faster_whisper_large_v3",),
    ),
]
RESULT_FILENAMES = {
    "results.csv",
    "results.json",
    "readability_results.csv",
    "readability_results.json",
    "readability_summary.md",
    "run_summary.md",
    "diarization_segments.csv",
    "diarization_segments.json",
    "separation_segments.csv",
    "separation_segments.json",
}
ROW_FIELDS = [
    "sample_id",
    "overlap_level",
    "ovr",
    "ovr_source",
    "pipeline",
    "model",
    "cer",
    "wer",
    "bert_precision",
    "bert_recall",
    "bert_f1",
    "bert_f2",
    "speaker_block_cer",
    "speaker_consistency",
    "trs_text",
    "trs_speaker",
    "runtime_seconds",
    "error",
]
SUMMARY_FIELDS = [
    "model",
    "pipeline",
    "runs",
    "errors",
    "avg_cer",
    "avg_wer",
    "avg_bert_f2",
    "avg_trs_text",
    "avg_speaker_block_cer",
    "avg_speaker_consistency",
    "avg_trs_speaker",
    "avg_runtime_seconds",
]
MODEL_SUMMARY_FIELDS = [
    "model",
    "runs",
    "errors",
    "avg_cer",
    "avg_wer",
    "avg_bert_f2",
    "avg_trs_text",
    "avg_speaker_block_cer",
    "avg_speaker_consistency",
    "avg_trs_speaker",
    "avg_runtime_seconds",
]


def main() -> int:
    args = parse_args()
    root = Path(args.output_root).expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"Output root does not exist: {root}")

    direct_rows: list[dict[str, str]] = []
    for _, target_name, legacy_names in MODEL_SPECS:
        target_dir = root / target_name
        source_dir = first_existing([target_dir, *(root / name for name in legacy_names)])
        if source_dir is None:
            print(f"skip missing model output: {target_dir}")
            continue
        if source_dir != target_dir:
            copy_result_files(source_dir, target_dir)
        rows = read_csv(source_dir / "readability_results.csv")
        write_model_outputs(target_dir, rows)
        direct_rows.extend(row for row in rows if row.get("pipeline") == "direct_asr")

    direct_dir = root / "direct_asr_benchmark"
    write_benchmark_outputs(
        direct_dir,
        direct_rows,
        title="Direct ASR Benchmark",
        report_name="direct_asr_selection_report.md",
        copy_figures_from=root / "asr_benchmark" / "direct_asr_figures",
    )

    print(f"Wrote {direct_dir}")
    for _, target_name, _ in MODEL_SPECS:
        target_dir = root / target_name
        if target_dir.exists():
            print(f"Wrote {target_dir}")
    return 0

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        default="sever_outputs",
        help="Root directory containing AliMeeting model outputs.",
    )
    return parser.parse_args()



def first_existing(paths) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None

def copy_result_files(source_dir: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(source_dir.iterdir()):
        if path.is_file() and path.name in RESULT_FILENAMES:
            shutil.copy2(path, target_dir / path.name)


def write_model_outputs(target_dir: Path, rows: list[dict[str, str]]) -> None:
    write_csv(target_dir / "readability_results_all.csv", rows, ROW_FIELDS)
    write_csv(target_dir / "model_pipeline_summary.csv", pipeline_summary(rows), SUMMARY_FIELDS)
    write_csv(target_dir / "model_summary.csv", model_summary(rows), MODEL_SUMMARY_FIELDS)
    (target_dir / "pipeline_selection_report.md").write_text(
        render_report("AliMeeting Model Pipeline Results", rows),
        encoding="utf-8",
    )


def write_benchmark_outputs(
    target_dir: Path,
    rows: list[dict[str, str]],
    title: str,
    report_name: str,
    copy_figures_from: Path | None = None,
) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    write_csv(target_dir / "readability_results_all.csv", rows, ROW_FIELDS)
    write_csv(target_dir / "model_pipeline_summary.csv", pipeline_summary(rows), SUMMARY_FIELDS)
    write_csv(target_dir / "model_summary.csv", model_summary(rows), MODEL_SUMMARY_FIELDS)
    (target_dir / report_name).write_text(render_report(title, rows), encoding="utf-8")
    if copy_figures_from and copy_figures_from.exists():
        figure_dir = target_dir / "figures"
        figure_dir.mkdir(parents=True, exist_ok=True)
        for path in sorted(copy_figures_from.rglob("*")):
            if path.is_file():
                rel = path.relative_to(copy_figures_from)
                destination = figure_dir / rel
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def pipeline_summary(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    summary = []
    keys = sorted({(row.get("model", ""), row.get("pipeline", "")) for row in rows})
    for model, pipeline in keys:
        group = [row for row in rows if row.get("model") == model and row.get("pipeline") == pipeline]
        clean = [row for row in group if not row.get("error")]
        summary.append(summary_row(clean, errors=len(group) - len(clean), model=model, pipeline=pipeline))
    return summary


def model_summary(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    summary = []
    for model in sorted({row.get("model", "") for row in rows}):
        group = [row for row in rows if row.get("model") == model]
        clean = [row for row in group if not row.get("error")]
        row = summary_row(clean, errors=len(group) - len(clean), model=model, pipeline=None)
        row.pop("pipeline", None)
        summary.append(row)
    return summary


def summary_row(
    rows: list[dict[str, str]],
    *,
    errors: int,
    model: str,
    pipeline: str | None,
) -> dict[str, object]:
    row = {
        "model": model,
        "runs": len(rows),
        "errors": errors,
        "avg_cer": avg(rows, "cer"),
        "avg_wer": avg(rows, "wer"),
        "avg_bert_f2": avg(rows, "bert_f2"),
        "avg_trs_text": avg(rows, "trs_text"),
        "avg_speaker_block_cer": avg(rows, "speaker_block_cer"),
        "avg_speaker_consistency": avg(rows, "speaker_consistency"),
        "avg_trs_speaker": avg(rows, "trs_speaker"),
        "avg_runtime_seconds": avg(rows, "runtime_seconds"),
    }
    if pipeline is not None:
        row["pipeline"] = pipeline
    return row


def avg(rows: list[dict[str, str]], field: str) -> float | None:
    values = [to_float(row.get(field)) for row in rows]
    values = [value for value in values if value is not None]
    return mean(values) if values else None


def to_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt(value: object) -> str:
    number = to_float(value)
    return "" if number is None else f"{number:.4f}"


def render_report(title: str, rows: list[dict[str, str]]) -> str:
    lines = [
        f"# {title}",
        "",
        "## Model/Pipeline Summary",
        "",
        "| Model | Pipeline | Runs | Errors | Avg CER | Avg WER | Avg TRS Text | Avg TRS Speaker | Avg Runtime |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in pipeline_summary(rows):
        lines.append(
            f"| {row['model']} | {row['pipeline']} | {row['runs']} | {row['errors']} | "
            f"{fmt(row['avg_cer'])} | {fmt(row['avg_wer'])} | {fmt(row['avg_trs_text'])} | "
            f"{fmt(row['avg_trs_speaker'])} | {fmt(row['avg_runtime_seconds'])}s |"
        )
    lines.extend(
        [
            "",
            "## Model Summary",
            "",
            "| Model | Runs | Errors | Avg CER | Avg WER | Avg TRS Text | Avg TRS Speaker | Avg Runtime |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in model_summary(rows):
        lines.append(
            f"| {row['model']} | {row['runs']} | {row['errors']} | "
            f"{fmt(row['avg_cer'])} | {fmt(row['avg_wer'])} | {fmt(row['avg_trs_text'])} | "
            f"{fmt(row['avg_trs_speaker'])} | {fmt(row['avg_runtime_seconds'])}s |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `readability_results_all.csv`: sample-level rows.",
            "- `model_pipeline_summary.csv`: model and pipeline aggregate table.",
            "- `model_summary.csv`: model aggregate table.",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
