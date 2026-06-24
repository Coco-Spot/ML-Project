#!/usr/bin/env python3
"""Generate pipeline comparison plots for a selected ASR model.

The script is independent from training and evaluation: it reads result CSV files, filters the selected model when requested, and writes figures only.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean
from typing import Iterable


DEFAULT_MODEL = "faster-whisper:large-v3"
DEFAULT_INPUT = "sever_outputs/alimeeting_faster_whisper_large_v3/readability_results.csv"
DEFAULT_OUTPUT_DIR = "sever_outputs/alimeeting_faster_whisper_large_v3/figures"
PIPELINE_ORDER = [
    "direct_asr",
    "diarization_asr",
    "diarization_turn_asr",
    "separation_asr",
    "llm_rag_refine",
]


def main() -> int:
    args = parse_args()
    rows = read_csv(Path(args.input))
    summary_rows = read_csv(Path(args.summary)) if args.summary else []
    rows = filter_model(rows, args.model)
    summary_rows = filter_model(summary_rows, args.model)
    if not rows and not summary_rows:
        raise SystemExit(
            "No rows found. Run evaluation first, or pass --input/--summary for an existing result CSV."
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_cer_by_pipeline_overlap(plt, rows, output_dir / "cer_by_pipeline_overlap.png", args.model)
    plot_metric_by_pipeline(
        plt,
        rows,
        summary_rows,
        output_dir / "trs_text_by_pipeline.png",
        args.model,
        metric="trs_text",
        summary_metric="avg_trs_text",
        title="TRS Text by Pipeline",
        ylabel="Average TRS Text",
        higher_is_better=True,
    )
    plot_metric_by_pipeline(
        plt,
        rows,
        summary_rows,
        output_dir / "trs_speaker_by_pipeline.png",
        args.model,
        metric="trs_speaker",
        summary_metric="avg_trs_speaker",
        title="TRS Speaker by Pipeline",
        ylabel="Average TRS Speaker",
        higher_is_better=True,
    )
    plot_metric_by_pipeline(
        plt,
        rows,
        summary_rows,
        output_dir / "runtime_by_pipeline.png",
        args.model,
        metric="runtime_seconds",
        summary_metric="avg_runtime_seconds",
        title="Runtime by Pipeline",
        ylabel="Average runtime (seconds)",
        higher_is_better=False,
    )

    print(f"Wrote {output_dir / 'cer_by_pipeline_overlap.png'}")
    print(f"Wrote {output_dir / 'trs_text_by_pipeline.png'}")
    print(f"Wrote {output_dir / 'trs_speaker_by_pipeline.png'}")
    print(f"Wrote {output_dir / 'runtime_by_pipeline.png'}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help="Row-level readability CSV. Defaults to the faster-whisper model result directory.",
    )
    parser.add_argument(
        "--summary",
        default="",
        help="Optional model/pipeline summary CSV. If omitted, metrics are averaged from --input.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where PNG figures are written.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Model to plot when the input contains multiple models. Use an empty string to keep all rows.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def filter_model(rows: list[dict[str, str]], model: str) -> list[dict[str, str]]:
    if not model:
        return rows
    return [row for row in rows if row.get("model") in {"", model}]


def to_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def clean_rows(rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if not row.get("error")]


def avg(values: Iterable[float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    return mean(clean) if clean else None


def ordered_pipelines(rows: Iterable[dict[str, str]]) -> list[str]:
    present = {row.get("pipeline", "") for row in rows if row.get("pipeline")}
    ordered = [pipeline for pipeline in PIPELINE_ORDER if pipeline in present]
    ordered.extend(sorted(present.difference(ordered)))
    return ordered


def overlap_levels(rows: Iterable[dict[str, str]]) -> list[str]:
    preferred = ["low", "medium", "high", "unknown"]
    present = {row.get("overlap_level", "") for row in rows if row.get("overlap_level")}
    ordered = [level for level in preferred if level in present]
    ordered.extend(sorted(present.difference(ordered)))
    return ordered


def display_pipeline(pipeline: str) -> str:
    return pipeline.replace("_", "\n")


def display_model(model: str) -> str:
    return model or "all models"


def save_or_note(plt, output: Path, title: str, message: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axis("off")
    ax.text(0.5, 0.55, title, ha="center", va="center", fontsize=15, weight="bold")
    ax.text(0.5, 0.42, message, ha="center", va="center", fontsize=11)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_cer_by_pipeline_overlap(
    plt,
    rows: list[dict[str, str]],
    output: Path,
    model: str,
) -> None:
    rows = clean_rows(rows)
    pipelines = ordered_pipelines(rows)
    levels = overlap_levels(rows)
    if not rows or not pipelines or not levels:
        save_or_note(plt, output, "CER by Pipeline and Overlap", "No CER/overlap rows available.")
        return

    fig, ax = plt.subplots(figsize=(max(10, 1.4 * len(levels) * len(pipelines)), 5.6))
    width = 0.8 / max(1, len(pipelines))
    xs = list(range(len(levels)))
    for index, pipeline in enumerate(pipelines):
        values = []
        for level in levels:
            group = [
                row for row in rows
                if row.get("pipeline") == pipeline and row.get("overlap_level") == level
            ]
            values.append(avg(to_float(row.get("cer")) for row in group))
        offsets = [x - 0.4 + width / 2 + index * width for x in xs]
        ax.bar(offsets, [value if value is not None else 0 for value in values], width=width, label=pipeline)

    ax.set_title(f"CER by Pipeline and Overlap ({display_model(model)})")
    ax.set_xlabel("Overlap level")
    ax.set_ylabel("Average CER")
    ax.set_xticks(xs, levels)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncols=min(3, len(pipelines)))
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_metric_by_pipeline(
    plt,
    rows: list[dict[str, str]],
    summary_rows: list[dict[str, str]],
    output: Path,
    model: str,
    *,
    metric: str,
    summary_metric: str,
    title: str,
    ylabel: str,
    higher_is_better: bool,
) -> None:
    source = clean_rows(summary_rows) or clean_rows(rows)
    pipelines = ordered_pipelines(source)
    if not source or not pipelines:
        save_or_note(plt, output, title, f"No {metric} rows available.")
        return

    field = summary_metric if summary_rows else metric
    values = []
    for pipeline in pipelines:
        group = [row for row in source if row.get("pipeline") == pipeline]
        values.append(avg(to_float(row.get(field)) for row in group))

    fig, ax = plt.subplots(figsize=(max(8.5, 1.4 * len(pipelines)), 5.2))
    bars = ax.bar([display_pipeline(pipeline) for pipeline in pipelines], [value if value is not None else 0 for value in values])
    ax.set_title(f"{title} ({display_model(model)})")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)

    valid_values = [value for value in values if value is not None]
    best_value = None
    if valid_values:
        best_value = max(valid_values) if higher_is_better else min(valid_values)
    for bar, value in zip(bars, values):
        if value is None:
            continue
        if value == best_value:
            bar.set_alpha(1.0)
        else:
            bar.set_alpha(0.72)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
