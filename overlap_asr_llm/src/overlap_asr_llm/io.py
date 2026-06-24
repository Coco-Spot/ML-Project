"""Output helpers."""

from __future__ import annotations

import csv
from html import escape
import json
import os
from pathlib import Path

from .pipelines import PipelineResult


FIELDNAMES = [
    "sample_id",
    "audio_path",
    "overlap_level",
    "pipeline",
    "model",
    "text",
    "speaker_labels",
    "runtime_seconds",
    "cer",
    "wer",
    "text_cer",
    "text_wer",
    "flat_cer",
    "flat_wer",
    "timeline_cer",
    "timeline_wer",
    "speaker_block_cer",
    "speaker_block_wer",
    "best_speaker_mapping",
    "score_basis",
    "error",
    "segments",
]


def write_results(
    results: list[PipelineResult],
    output_dir: Path,
    base_dir: Path | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [_public_result_row(result, base_dir) for result in results]

    with (output_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    with (output_dir / "results.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(_csv_safe_rows(rows))

    write_summary(results, output_dir / "run_summary.md")
    if any(_is_diarization_pipeline(result.pipeline) for result in results):
        write_diarization_segments(results, output_dir, base_dir)
    else:
        _remove_stale_outputs(
            output_dir,
            [
                "diarization_segments.csv",
                "diarization_segments.json",
            ],
        )
    if any(result.pipeline == "llm_rag_refine" and result.segments for result in results):
        write_llm_rag_source_segments(results, output_dir, base_dir)
    else:
        _remove_stale_outputs(
            output_dir,
            [
                "llm_rag_source_segments.csv",
                "llm_rag_source_segments.json",
            ],
        )
    if any(result.pipeline == "separation_asr" for result in results):
        write_separation_segments(results, output_dir, base_dir)
    else:
        _remove_stale_outputs(
            output_dir,
            [
                "separation_segments.csv",
                "separation_segments.json",
            ],
        )


def _remove_stale_outputs(output_dir: Path, filenames: list[str]) -> None:
    for filename in filenames:
        path = output_dir / filename
        if path.exists():
            path.unlink()


def _is_diarization_pipeline(pipeline: str) -> bool:
    return pipeline in {"diarization_asr", "diarization_turn_asr"}


def _public_path(path: str, base_dir: Path | None) -> str:
    path_obj = Path(path)
    if base_dir is None:
        return path
    try:
        return path_obj.relative_to(base_dir).as_posix()
    except ValueError:
        try:
            parent_relative = path_obj.relative_to(base_dir.parent).as_posix()
            return f"../{parent_relative}"
        except ValueError:
            return path_obj.name


def _public_result_row(
    result: PipelineResult,
    base_dir: Path | None,
) -> dict[str, object]:
    row = result.to_dict()
    row["audio_path"] = _public_path(str(row["audio_path"]), base_dir)
    row["segments"] = _public_segments(row.get("segments", []), base_dir)
    return row


def _public_segments(
    segments: object,
    base_dir: Path | None,
) -> list[dict[str, object]]:
    public_segments = []
    if not isinstance(segments, list):
        return public_segments
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        public_segment = dict(segment)
        if "separated_audio_path" in public_segment:
            public_segment["separated_audio_path"] = _public_path(
                str(public_segment["separated_audio_path"]),
                base_dir,
            )
        public_segments.append(public_segment)
    return public_segments


def _csv_safe_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    csv_rows = []
    for row in rows:
        csv_row = dict(row)
        csv_row["segments"] = json.dumps(csv_row.get("segments", []), ensure_ascii=False)
        csv_rows.append(csv_row)
    return csv_rows


def write_diarization_segments(
    results: list[PipelineResult],
    output_dir: Path,
    base_dir: Path | None = None,
) -> None:
    rows = []
    for result in results:
        if not _is_diarization_pipeline(result.pipeline) or result.error:
            continue
        for index, segment in enumerate(result.segments, start=1):
            rows.append(
                {
                    "sample_id": result.sample_id,
                    "audio_path": _public_path(result.audio_path, base_dir),
                    "overlap_level": result.overlap_level,
                    "pipeline": result.pipeline,
                    "model": result.model,
                    "segment_index": index,
                    "start": segment.get("start", ""),
                    "end": segment.get("end", ""),
                    "speaker": segment.get("speaker", "UNKNOWN"),
                    "text": segment.get("text", ""),
                }
            )
    _write_segment_rows(rows, output_dir, "diarization_segments")


def write_llm_rag_source_segments(
    results: list[PipelineResult],
    output_dir: Path,
    base_dir: Path | None = None,
) -> None:
    rows = []
    for result in results:
        if result.pipeline != "llm_rag_refine" or result.error:
            continue
        for index, segment in enumerate(result.segments, start=1):
            rows.append(
                {
                    "sample_id": result.sample_id,
                    "audio_path": _public_path(result.audio_path, base_dir),
                    "overlap_level": result.overlap_level,
                    "pipeline": result.pipeline,
                    "model": result.model,
                    "segment_index": index,
                    "start": segment.get("start", ""),
                    "end": segment.get("end", ""),
                    "speaker": segment.get("speaker", "UNKNOWN"),
                    "text": segment.get("text", ""),
                }
            )
    _write_segment_rows(rows, output_dir, "llm_rag_source_segments")


def _write_segment_rows(
    rows: list[dict[str, object]],
    output_dir: Path,
    stem: str,
) -> None:
    fieldnames = [
        "sample_id",
        "audio_path",
        "overlap_level",
        "pipeline",
        "model",
        "segment_index",
        "start",
        "end",
        "speaker",
        "text",
    ]
    with (output_dir / f"{stem}.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    with (output_dir / f"{stem}.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_separation_segments(
    results: list[PipelineResult],
    output_dir: Path,
    base_dir: Path | None = None,
) -> None:
    rows = []
    for result in results:
        if result.pipeline != "separation_asr" or result.error:
            continue
        for segment in result.segments:
            rows.append(
                {
                    "sample_id": result.sample_id,
                    "audio_path": _public_path(result.audio_path, base_dir),
                    "overlap_level": result.overlap_level,
                    "source_index": segment.get("source_index", ""),
                    "separated_audio_path": _public_path(
                        str(segment.get("separated_audio_path", "")),
                        base_dir,
                    ),
                    "start": segment.get("start", ""),
                    "end": segment.get("end", ""),
                    "speaker": segment.get("speaker", "UNKNOWN"),
                    "text": segment.get("text", ""),
                }
            )

    with (output_dir / "separation_segments.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    fieldnames = [
        "sample_id",
        "audio_path",
        "overlap_level",
        "source_index",
        "separated_audio_path",
        "start",
        "end",
        "speaker",
        "text",
    ]
    with (output_dir / "separation_segments.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(results: list[PipelineResult], path: Path) -> None:
    html_path = path.with_suffix(".html")
    write_html = os.environ.get("OVERLAP_ASR_LLM_WRITE_HTML_SUMMARY") == "1"
    if not write_html:
        html_path.unlink(missing_ok=True)

    if results and all(result.pipeline == "diarization_asr" for result in results):
        write_diarization_summary(results, path)
        return
    if results and all(result.pipeline == "separation_asr" for result in results):
        write_separation_summary(results, path)
        return

    lines = _comparison_summary_lines(results)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if write_html:
        _write_summary_html(results, html_path)


def _comparison_summary_lines(results: list[PipelineResult]) -> list[str]:
    lines = [
        "# Run Summary",
        "",
        "## Pipeline Ranking",
        "",
        "| Rank | Pipeline | Runs | Avg CER | Avg WER | Avg Runtime | Wins | Errors |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rank, row in enumerate(_pipeline_ranking(results), start=1):
        lines.append(
            "| "
            f"{rank} | "
            f"{row['label']} | "
            f"{row['runs']} | "
            f"{_metric(row['avg_cer'])} | "
            f"{_metric(row['avg_wer'])} | "
            f"{row['avg_runtime']:.2f}s | "
            f"{row['wins']} | "
            f"{row['errors']} |"
        )

    lines.extend(
        [
            "",
            "## Best By Overlap",
            "",
            "| Sample | Overlap | Best Pipeline | CER | WER | Runtime | Runner-Up | Delta CER |",
            "| --- | --- | --- | ---: | ---: | ---: | --- | ---: |",
        ]
    )
    for row in _best_by_sample(results):
        lines.append(
            "| "
            f"{row['sample_id']} | "
            f"{row['overlap_level']} | "
            f"{row['best_label']} | "
            f"{_metric(row['cer'])} | "
            f"{_metric(row['wer'])} | "
            f"{row['runtime']:.2f}s | "
            f"{row['runner_up_label']} | "
            f"{_metric(row['delta_cer'])} |"
        )

    ablation = _diarization_ablation(results)
    if ablation:
        lines.extend(
            [
                "",
                "## Diarization Order Ablation",
                "",
                "| Sample | Overlap | Full-Audio Align CER/WER | Turn-Level ASR CER/WER | CER Change | Runtime Change | Better |",
                "| --- | --- | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for row in ablation:
            lines.append(
                "| "
                f"{row['sample_id']} | "
                f"{row['overlap_level']} | "
                f"{_metric(row['full_cer'])}/{_metric(row['full_wer'])} | "
                f"{_metric(row['turn_cer'])}/{_metric(row['turn_wer'])} | "
                f"{_signed_metric(row['cer_change'])} | "
                f"{_signed_seconds(row['runtime_change'])} | "
                f"{row['better']} |"
            )

    lines.extend(
        [
            "",
            "## Detailed Results",
            "",
            "| Sample | Overlap | Pipeline | CER | WER | Runtime | Basis | Error |",
            "| --- | --- | --- | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for result in results:
        lines.append(
            "| "
            f"{result.sample_id} | "
            f"{result.overlap_level} | "
            f"{_result_label(result, results)} | "
            f"{_metric(result.cer)} | "
            f"{_metric(result.wer)} | "
            f"{result.runtime_seconds:.2f}s | "
            f"{result.score_basis} | "
            f"{result.error or ''} |"
        )

    return lines


def _valid_results(results: list[PipelineResult]) -> list[PipelineResult]:
    return [
        result
        for result in results
        if not result.error and result.cer is not None and result.wer is not None
    ]


def _pipeline_ranking(results: list[PipelineResult]) -> list[dict[str, object]]:
    valid = _valid_results(results)
    wins = {}
    for row in _best_by_sample(results):
        wins[row["best_label"]] = wins.get(row["best_label"], 0) + 1

    rows = []
    labels = dict.fromkeys(_result_label(result, results) for result in results)
    for label in labels:
        pipeline_results = [
            result for result in results if _result_label(result, results) == label
        ]
        ok = [result for result in valid if _result_label(result, results) == label]
        rows.append(
            {
                "label": label,
                "runs": len(pipeline_results),
                "avg_cer": _average(result.cer for result in ok),
                "avg_wer": _average(result.wer for result in ok),
                "avg_runtime": _average(result.runtime_seconds for result in pipeline_results) or 0.0,
                "wins": wins.get(label, 0),
                "errors": sum(1 for result in pipeline_results if result.error),
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            float("inf") if row["avg_cer"] is None else row["avg_cer"],
            float("inf") if row["avg_wer"] is None else row["avg_wer"],
            row["avg_runtime"],
        ),
    )


def _best_by_sample(results: list[PipelineResult]) -> list[dict[str, object]]:
    rows = []
    sample_ids = dict.fromkeys(result.sample_id for result in results)
    for sample_id in sample_ids:
        candidates = [
            result
            for result in _valid_results(results)
            if result.sample_id == sample_id
        ]
        if not candidates:
            continue
        ordered = sorted(candidates, key=lambda result: (result.cer, result.wer, result.runtime_seconds))
        best = ordered[0]
        runner_up = ordered[1] if len(ordered) > 1 else None
        rows.append(
            {
                "sample_id": best.sample_id,
                "overlap_level": best.overlap_level,
                "best_label": _result_label(best, results),
                "cer": best.cer,
                "wer": best.wer,
                "runtime": best.runtime_seconds,
                "runner_up_label": "" if runner_up is None else _result_label(runner_up, results),
                "delta_cer": None if runner_up is None else runner_up.cer - best.cer,
            }
        )
    return rows


def _diarization_ablation(results: list[PipelineResult]) -> list[dict[str, object]]:
    rows = []
    sample_ids = dict.fromkeys(result.sample_id for result in results)
    for sample_id in sample_ids:
        full = _find_result(results, sample_id, "diarization_asr")
        turn = _find_result(results, sample_id, "diarization_turn_asr")
        if full is None or turn is None:
            continue
        if full.cer is None or turn.cer is None:
            continue
        better = "tie"
        if full.cer < turn.cer:
            better = "full_audio_align"
        elif turn.cer < full.cer:
            better = "turn_level_asr"
        rows.append(
            {
                "sample_id": sample_id,
                "overlap_level": full.overlap_level,
                "full_cer": full.cer,
                "full_wer": full.wer,
                "turn_cer": turn.cer,
                "turn_wer": turn.wer,
                "cer_change": turn.cer - full.cer,
                "runtime_change": turn.runtime_seconds - full.runtime_seconds,
                "better": better,
            }
        )
    return rows


def _find_result(
    results: list[PipelineResult],
    sample_id: str,
    pipeline: str,
) -> PipelineResult | None:
    for result in results:
        if result.sample_id == sample_id and result.pipeline == pipeline and not result.error:
            return result
    return None


def _result_label(result: PipelineResult, all_results: list[PipelineResult]) -> str:
    models = {
        item.model
        for item in all_results
        if item.pipeline == result.pipeline
    }
    if len(models) <= 1:
        return result.pipeline
    return f"{result.pipeline} ({_short_model_label(result.model)})"


def _short_model_label(model: str) -> str:
    if "<-" in model:
        source = model.split("<-", 1)[1].split(":", 1)[0]
        return source
    if "+" in model:
        return model.rsplit("+", 1)[-1]
    return model


def _average(values) -> float | None:
    items = [value for value in values if value is not None]
    if not items:
        return None
    return sum(items) / len(items)


def _metric(value: object) -> str:
    if value is None:
        return ""
    return f"{float(value):.4f}"


def _signed_metric(value: object) -> str:
    if value is None:
        return ""
    return f"{float(value):+.4f}"


def _signed_seconds(value: object) -> str:
    if value is None:
        return ""
    return f"{float(value):+.2f}s"


def _write_summary_html(results: list[PipelineResult], path: Path) -> None:
    md_lines = _comparison_summary_lines(results)
    html_lines = [
        "<!doctype html>",
        "<html><head><meta charset=\"utf-8\">",
        "<title>Run Summary</title>",
        "<style>",
        "body{font-family:system-ui,-apple-system,Segoe UI,sans-serif;margin:32px;line-height:1.4;color:#1f2933}",
        "h1{font-size:28px;margin-bottom:20px} h2{font-size:20px;margin-top:28px}",
        "table{border-collapse:collapse;width:100%;margin:12px 0 24px;font-size:14px}",
        "th,td{border:1px solid #d8dee4;padding:6px 8px;text-align:left;vertical-align:top}",
        "th{background:#f3f4f6;font-weight:650} td:nth-child(n+4){text-align:right}",
        "tr:nth-child(even) td{background:#fafafa}",
        "</style></head><body>",
    ]
    index = 0
    while index < len(md_lines):
        line = md_lines[index]
        if line.startswith("# "):
            html_lines.append(f"<h1>{escape(line[2:])}</h1>")
            index += 1
        elif line.startswith("## "):
            html_lines.append(f"<h2>{escape(line[3:])}</h2>")
            index += 1
        elif line.startswith("| "):
            table_lines = []
            while index < len(md_lines) and md_lines[index].startswith("| "):
                table_lines.append(md_lines[index])
                index += 1
            html_lines.extend(_markdown_table_to_html(table_lines))
        else:
            index += 1
    html_lines.append("</body></html>")
    path.write_text("\n".join(html_lines) + "\n", encoding="utf-8")


def _markdown_table_to_html(lines: list[str]) -> list[str]:
    if len(lines) < 2:
        return []
    headers = _split_markdown_row(lines[0])
    body = [_split_markdown_row(line) for line in lines[2:]]
    html = ["<table>", "<thead><tr>"]
    html.extend(f"<th>{escape(cell)}</th>" for cell in headers)
    html.append("</tr></thead><tbody>")
    for row in body:
        html.append("<tr>")
        html.extend(f"<td>{escape(cell)}</td>" for cell in row)
        html.append("</tr>")
    html.append("</tbody></table>")
    return html


def _split_markdown_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def write_separation_summary(results: list[PipelineResult], path: Path) -> None:
    lines = [
        "# Run Summary",
        "",
        "| Sample | Overlap | Pipeline | Model | Separated Sources | Sources With Text | Score Basis | Primary CER | Primary WER | Speaker CER | Speaker WER | Timeline CER | Timeline WER | Runtime | Error |",
        "| --- | --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for result in results:
        sources = {
            str(segment.get("source_index", segment.get("speaker", "")))
            for segment in result.segments
            if segment.get("source_index", segment.get("speaker", ""))
        }
        sources_with_text = {
            str(segment.get("source_index", segment.get("speaker", "")))
            for segment in result.segments
            if segment.get("source_index", segment.get("speaker", ""))
            and str(segment.get("text", "")).strip()
        }
        cer_value = "" if result.cer is None else f"{result.cer:.4f}"
        wer_value = "" if result.wer is None else f"{result.wer:.4f}"
        speaker_cer = (
            "" if result.speaker_block_cer is None else f"{result.speaker_block_cer:.4f}"
        )
        speaker_wer = (
            "" if result.speaker_block_wer is None else f"{result.speaker_block_wer:.4f}"
        )
        timeline_cer = (
            "" if result.timeline_cer is None else f"{result.timeline_cer:.4f}"
        )
        timeline_wer = (
            "" if result.timeline_wer is None else f"{result.timeline_wer:.4f}"
        )
        lines.append(
            "| "
            f"{result.sample_id} | "
            f"{result.overlap_level} | "
            f"{result.pipeline} | "
            f"{result.model} | "
            f"{len(sources)} | "
            f"{len(sources_with_text)} | "
            f"{result.score_basis} | "
            f"{cer_value} | "
            f"{wer_value} | "
            f"{speaker_cer} | "
            f"{speaker_wer} | "
            f"{timeline_cer} | "
            f"{timeline_wer} | "
            f"{result.runtime_seconds:.4f} | "
            f"{result.error or ''} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_diarization_summary(results: list[PipelineResult], path: Path) -> None:
    lines = [
        "# Run Summary",
        "",
        "| Sample | Overlap | Pipeline | Model | Speakers | Segments | Segments With Text | Score Basis | Primary CER | Primary WER | Speaker CER | Speaker WER | Timeline CER | Timeline WER | Runtime | Error |",
        "| --- | --- | --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for result in results:
        speakers = ",".join(
            dict.fromkeys(str(segment.get("speaker", "UNKNOWN")) for segment in result.segments)
        )
        segments_with_text = sum(
            1 for segment in result.segments if str(segment.get("text", "")).strip()
        )
        cer_value = "" if result.cer is None else f"{result.cer:.4f}"
        wer_value = "" if result.wer is None else f"{result.wer:.4f}"
        speaker_cer = (
            "" if result.speaker_block_cer is None else f"{result.speaker_block_cer:.4f}"
        )
        speaker_wer = (
            "" if result.speaker_block_wer is None else f"{result.speaker_block_wer:.4f}"
        )
        timeline_cer = (
            "" if result.timeline_cer is None else f"{result.timeline_cer:.4f}"
        )
        timeline_wer = (
            "" if result.timeline_wer is None else f"{result.timeline_wer:.4f}"
        )
        lines.append(
            "| "
            f"{result.sample_id} | "
            f"{result.overlap_level} | "
            f"{result.pipeline} | "
            f"{result.model} | "
            f"{speakers} | "
            f"{len(result.segments)} | "
            f"{segments_with_text} | "
            f"{result.score_basis} | "
            f"{cer_value} | "
            f"{wer_value} | "
            f"{speaker_cer} | "
            f"{speaker_wer} | "
            f"{timeline_cer} | "
            f"{timeline_wer} | "
            f"{result.runtime_seconds:.4f} | "
            f"{result.error or ''} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
