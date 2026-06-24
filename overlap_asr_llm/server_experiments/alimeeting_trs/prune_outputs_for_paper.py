#!/usr/bin/env python3
"""Prune server outputs down to paper-facing artifacts.

Default mode is a dry run. Pass --apply to delete files and directories that are
not needed for paper tables, reports, or figures. Model caches are never deleted
unless --include-caches is passed.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil


KEEP_FILENAMES = {
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
    "llm_rag_source_segments.csv",
    "llm_rag_source_segments.json",
    "readability_results_all.csv",
    "model_pipeline_summary.csv",
    "model_summary.csv",
    "alimeeting_trs_selection_report.md",
    "direct_asr_selection_report.md",
    "cer_by_model_overlap.png",
    "trs_text_by_model.png",
    "trs_speaker_heatmap.png",
    "runtime_by_model.png",
    "runtime_by_pipeline.png",
    "trs_speaker_by_pipeline.png",
    "trs_text_by_pipeline.png",
    "cer_by_pipeline_overlap.png",
}
KEEP_DIR_NAMES = {"figures"}
KEEP_FIGURE_SUFFIXES = {".csv", ".pdf", ".png", ".svg", ".tif", ".tiff"}
DROP_DIR_NAMES = {"separated_audio", "logs", "mock", "asr_benchmark_old"}
CACHE_DIR_NAMES = {"caches"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        default="sever_outputs",
        help="Root output directory to prune.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually remove files. Without this flag, only print actions.",
    )
    parser.add_argument(
        "--include-caches",
        action="store_true",
        help="Also remove cache directories under the output root.",
    )
    return parser.parse_args()


def should_keep(path: Path, root: Path, include_caches: bool) -> bool:
    rel_parts = path.relative_to(root).parts
    if not include_caches and any(part in CACHE_DIR_NAMES for part in rel_parts):
        return True
    if path.is_dir():
        return path.name in KEEP_DIR_NAMES or "figures" in path.name
    if any("figures" in part for part in rel_parts) and path.suffix.lower() in KEEP_FIGURE_SUFFIXES:
        return True
    return path.name in KEEP_FILENAMES


def prune(root: Path, apply: bool, include_caches: bool) -> int:
    if not root.exists():
        raise SystemExit(f"Output root does not exist: {root}")

    removals: list[Path] = []
    for path in sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if not path.exists():
            continue
        rel_parts = path.relative_to(root).parts
        if any(part in DROP_DIR_NAMES for part in rel_parts):
            removals.append(path)
            continue
        if include_caches and any(part in CACHE_DIR_NAMES for part in rel_parts):
            removals.append(path)
            continue
        if path.is_file() and not should_keep(path, root, include_caches):
            removals.append(path)

    collapsed: list[Path] = []
    removal_set = set(removals)
    for path in removals:
        if any(parent in removal_set for parent in path.parents if parent != root.parent):
            continue
        collapsed.append(path)

    for path in sorted(collapsed):
        action = "remove dir" if path.is_dir() else "remove file"
        print(f"{action}: {path}")
        if apply:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink(missing_ok=True)

    print(("Removed" if apply else "Would remove") + f" {len(collapsed)} paths")
    return len(collapsed)


def main() -> int:
    args = parse_args()
    prune(Path(args.output_root).expanduser().resolve(), args.apply, args.include_caches)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
