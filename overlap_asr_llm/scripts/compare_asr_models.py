"""Run the same experiment with multiple ASR providers and summarize results."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from overlap_asr_llm.config import load_config  # noqa: E402
from overlap_asr_llm.io import FIELDNAMES, _public_path  # noqa: E402
from overlap_asr_llm.pipelines import PipelineResult, run_all  # noqa: E402


COMPARISON_FIELDNAMES = ["asr_model", *FIELDNAMES]


def format_metric(value: object) -> str:
    if value is None:
        return ""
    return f"{float(value):.4f}"


def write_report(
    results_by_model: dict[str, list[PipelineResult]],
    output_path: Path,
) -> None:
    lines = [
        "# Run Summary",
        "",
    ]
    for model_name, results in results_by_model.items():
        lines.extend(
            [
                f"## {model_name}",
                "",
                "| Sample | Overlap | Pipeline | Model | CER | WER | Runtime | Error |",
                "| --- | --- | --- | --- | ---: | ---: | ---: | --- |",
            ]
        )
        for result in results:
            lines.append(
                "| "
                f"{result.sample_id} | "
                f"{result.overlap_level} | "
                f"{result.pipeline} | "
                f"{result.model} | "
                f"{format_metric(result.cer)} | "
                f"{format_metric(result.wer)} | "
                f"{result.runtime_seconds:.4f} | "
                f"{result.error or ''} |"
            )
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def combined_rows(
    results_by_model: dict[str, list[PipelineResult]],
    base_dir: Path,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model_name, results in results_by_model.items():
        for result in results:
            row = {"asr_model": model_name, **result.to_dict()}
            row["audio_path"] = _public_path(str(row["audio_path"]), base_dir)
            rows.append(row)
    return rows


def write_results(
    results_by_model: dict[str, list[PipelineResult]],
    output_dir: Path,
    base_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = combined_rows(results_by_model, base_dir)

    with (output_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    with (output_dir / "results.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COMPARISON_FIELDNAMES)
        writer.writeheader()
        writer.writerows(csv_safe_rows(rows))


def csv_safe_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    csv_rows = []
    for row in rows:
        csv_row = dict(row)
        csv_row["segments"] = json.dumps(csv_row.get("segments", []), ensure_ascii=False)
        csv_rows.append(csv_row)
    return csv_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/direct_asr.json",
        help="Path to the base experiment config.",
    )
    parser.add_argument(
        "--whisper-model",
        default="base",
        help="Fallback Whisper model if config has no asr_models.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Override config asr_models from the command line.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Markdown report path. Defaults to <config output_dir>/run_summary.md.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_config(args.config)
    models = args.models or config.asr_models or ["funasr"]
    output_path = (
        Path(args.output).resolve()
        if args.output
        else config.output_dir / "run_summary.md"
    )

    results_by_model: dict[str, list[PipelineResult]] = {}
    for model_name in models:
        model_config = replace(
            config,
            models={**config.models, "asr": model_name},
        )
        results = run_all(model_config)
        results_by_model[model_name] = results
        print(f"Finished {model_name}")

    write_report(results_by_model, output_path)
    write_results(results_by_model, config.output_dir, config.base_dir)
    print(f"Wrote combined report to {output_path}")
    print(f"Wrote combined results to {config.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
