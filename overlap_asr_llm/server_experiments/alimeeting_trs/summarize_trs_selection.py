#!/usr/bin/env python3
"""Summarize CER/WER winners versus TRS Speaker winners."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean


DEFAULT_RESULTS = [
    "outputs/alimeeting_whisper_large_v3/readability_results.csv",
    "outputs/alimeeting_faster_whisper_large_v3/readability_results.csv",
    "outputs/alimeeting_funasr/readability_results.csv",
]
SUMMARY_FIELDS = [
    "model",
    "pipeline",
    "runs",
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
    rows = []
    for result_path in args.results:
        rows.extend(load_rows(Path(result_path)))
    if not rows:
        raise SystemExit("No readability rows found. Run evaluate first.")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_report(rows), encoding="utf-8")
    write_csv(Path(args.combined_csv), rows, rows[0].keys())
    write_csv(Path(args.summary_csv), model_pipeline_summary(rows), SUMMARY_FIELDS)
    write_csv(Path(args.model_summary_csv), model_summary(rows), MODEL_SUMMARY_FIELDS)
    print(f"Wrote {output}")
    print(f"Wrote {args.combined_csv}")
    print(f"Wrote {args.summary_csv}")
    print(f"Wrote {args.model_summary_csv}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", nargs="+", default=DEFAULT_RESULTS)
    parser.add_argument("--output", default="sever_outputs/asr_benchmark/alimeeting_trs_selection_report.md")
    parser.add_argument("--combined-csv", default="sever_outputs/asr_benchmark/readability_results_all.csv")
    parser.add_argument("--summary-csv", default="sever_outputs/asr_benchmark/model_pipeline_summary.csv")
    parser.add_argument("--model-summary-csv", default="sever_outputs/asr_benchmark/model_summary.csv")
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(fieldnames)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def render_report(rows: list[dict[str, str]]) -> str:
    lines = [
        "# AliMeeting ASR Benchmark Report",
        "",
        "## Pipeline Ranking By TRS Speaker",
        "",
        "| Rank | Model | Pipeline | Runs | Avg CER | Avg WER | Avg TRS Text | Avg TRS Speaker | Avg Runtime |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    ranking = model_pipeline_summary(rows)
    ranking.sort(key=lambda row: none_last_negative(to_float(row["avg_trs_speaker"])))
    for index, row in enumerate(ranking, start=1):
        lines.append(
            f"| {index} | {row['model']} | {row['pipeline']} | {row['runs']} | "
            f"{fmt(to_float(row['avg_cer']))} | {fmt(to_float(row['avg_wer']))} | "
            f"{fmt(to_float(row['avg_trs_text']))} | {fmt(to_float(row['avg_trs_speaker']))} | "
            f"{fmt(to_float(row['avg_runtime_seconds']))}s |"
        )

    lines.extend(
        [
            "",
            "## ASR Model Summary",
            "",
            "| Rank | Model | Runs | Avg CER | Avg WER | Avg TRS Text | Avg TRS Speaker | Avg Runtime |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    model_rows = model_summary(rows)
    model_rows.sort(key=lambda row: none_last_negative(to_float(row["avg_trs_text"])))
    for index, row in enumerate(model_rows, start=1):
        lines.append(
            f"| {index} | {row['model']} | {row['runs']} | "
            f"{fmt(to_float(row['avg_cer']))} | {fmt(to_float(row['avg_wer']))} | "
            f"{fmt(to_float(row['avg_trs_text']))} | {fmt(to_float(row['avg_trs_speaker']))} | "
            f"{fmt(to_float(row['avg_runtime_seconds']))}s |"
        )

    lines.extend(
        [
            "",
            "## Winner Conflicts",
            "",
            "| Sample | Model | OVR | Best CER Pipeline | CER | Best TRS Speaker Pipeline | TRS Speaker | Interpretation |",
            "| --- | --- | ---: | --- | ---: | --- | ---: | --- |",
        ]
    )
    conflicts = 0
    grouped_keys = sorted({(row["sample_id"], row["model"]) for row in rows})
    for sample_id, model in grouped_keys:
        group = [
            row for row in rows
            if row["sample_id"] == sample_id and row["model"] == model and not row.get("error")
        ]
        best_cer = best_min(group, "cer")
        best_trs = best_max(group, "trs_speaker")
        if not best_cer or not best_trs:
            continue
        if best_cer["pipeline"] == best_trs["pipeline"]:
            continue
        conflicts += 1
        interpretation = "TRS favors speaker-attributed usefulness over lower surface edit distance."
        lines.append(
            f"| {sample_id} | {model} | {fmt(to_float(best_trs.get('ovr')))} | "
            f"{best_cer['pipeline']} | {fmt(to_float(best_cer.get('cer')))} | "
            f"{best_trs['pipeline']} | {fmt(to_float(best_trs.get('trs_speaker')))} | "
            f"{interpretation} |"
        )
    if conflicts == 0:
        lines.append("| - | - | - | - | - | - | - | No CER/TRS Speaker winner conflicts found. |")

    lines.extend(
        [
            "",
            "## Paper Outputs",
            "",
            "- `sever_outputs/asr_benchmark/readability_results_all.csv`: row-level results.",
            "- `sever_outputs/asr_benchmark/model_pipeline_summary.csv`: model by pipeline summary.",
            "- `sever_outputs/asr_benchmark/model_summary.csv`: model-level summary.",
            "- `sever_outputs/asr_benchmark/figures/`: publication-style plots.",
        ]
    )
    return "\n".join(lines) + "\n"


def model_pipeline_summary(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    summary = []
    for key in sorted({(row["model"], row["pipeline"]) for row in rows}):
        group = clean_group(rows, model=key[0], pipeline=key[1])
        summary.append(summary_row(group, model=key[0], pipeline=key[1]))
    return summary


def model_summary(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    summary = []
    for model in sorted({row["model"] for row in rows}):
        group = clean_group(rows, model=model)
        row = summary_row(group, model=model, pipeline=None)
        row.pop("pipeline", None)
        summary.append(row)
    return summary


def clean_group(rows: list[dict[str, str]], model: str, pipeline: str | None = None) -> list[dict[str, str]]:
    group = [row for row in rows if row.get("model") == model and not row.get("error")]
    if pipeline is not None:
        group = [row for row in group if row.get("pipeline") == pipeline]
    return group


def summary_row(group: list[dict[str, str]], model: str, pipeline: str | None) -> dict[str, object]:
    row = {
        "model": model,
        "runs": len(group),
        "avg_cer": avg(group, "cer"),
        "avg_wer": avg(group, "wer"),
        "avg_bert_f2": avg(group, "bert_f2"),
        "avg_trs_text": avg(group, "trs_text"),
        "avg_speaker_block_cer": avg(group, "speaker_block_cer"),
        "avg_speaker_consistency": avg(group, "speaker_consistency"),
        "avg_trs_speaker": avg(group, "trs_speaker"),
        "avg_runtime_seconds": avg(group, "runtime_seconds"),
    }
    if pipeline is not None:
        row["pipeline"] = pipeline
    return row


def avg(rows: list[dict[str, str]], field: str) -> float | None:
    values = [to_float(row.get(field)) for row in rows]
    values = [value for value in values if value is not None]
    return mean(values) if values else None


def best_min(rows: list[dict[str, str]], field: str) -> dict[str, str] | None:
    candidates = [(to_float(row.get(field)), row) for row in rows]
    candidates = [(value, row) for value, row in candidates if value is not None]
    return min(candidates, key=lambda item: item[0])[1] if candidates else None


def best_max(rows: list[dict[str, str]], field: str) -> dict[str, str] | None:
    candidates = [(to_float(row.get(field)), row) for row in rows]
    candidates = [(value, row) for value, row in candidates if value is not None]
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def to_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def none_last_negative(value: float | None) -> tuple[bool, float]:
    return (value is None, 0.0 if value is None else -value)


def fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
