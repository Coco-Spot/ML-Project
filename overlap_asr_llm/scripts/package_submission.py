"""Create a safe zip archive for final submission.

The script packages source code, docs, configs, tests, sample audio, and selected
result files. It deliberately skips local secrets, virtual environments, model
caches, and temporary experiment outputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = ROOT.parent
ARCHIVE = ROOT / "overlap_asr_llm_project.zip"
EXTRA_PACKAGE_FILES = (
    REPOSITORY_ROOT / "README.md",
    REPOSITORY_ROOT / "Makefile",
)
EXTRA_PACKAGE_DIRS = (
    REPOSITORY_ROOT / "image",
)
EXTRA_PACKAGE_SUFFIXES = {
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".webp",
}
SKIP_PARTS = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".venv",
    "venv",
    "env",
    ".git",
    ".idea",
    ".vscode",
    "models",
    "checkpoints",
    "server_experiments",
}
SKIP_FILENAMES = {
    ".DS_Store",
    ".coverage",
    ".env",
    ARCHIVE.name,
}
SKIP_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".zip",
    ".egg-info",
}
SKIP_OUTPUT_DIRS = {
    "caches",
    "mock",
    "speaker_app",
}
KEEP_OUTPUT_DIRS = {
    "all_pipelines",
    "direct_asr",
    "diarization_asr",
    "separation_asr",
    "speaker_llm_pipeline",
}
SKIP_PREFIXES = {
    ROOT / "build",
    ROOT / "dist",
}


def _is_relative_to(path: Path, prefix: Path) -> bool:
    try:
        path.relative_to(prefix)
        return True
    except ValueError:
        return False


def _contains_skip_part(path: Path) -> bool:
    parts = set(path.relative_to(ROOT).parts)
    return (
        bool(parts & SKIP_PARTS)
        or any(part.startswith("pretrained_") for part in parts)
        or any(part.endswith(".egg-info") for part in parts)
    )


def _is_env_file(path: Path) -> bool:
    return path.name == ".env" or path.name.startswith(".env.")


def _is_output_file_to_skip(path: Path) -> bool:
    try:
        relative = path.relative_to(ROOT / "outputs")
    except ValueError:
        return False
    if path.suffix == ".html":
        return True
    if not relative.parts:
        return False
    output_dir = relative.parts[0]
    if output_dir in SKIP_OUTPUT_DIRS:
        return True
    return output_dir not in KEEP_OUTPUT_DIRS


def should_skip(path: Path) -> bool:
    if path == ROOT:
        return False
    if _contains_skip_part(path):
        return True
    if path.name in SKIP_FILENAMES or _is_env_file(path):
        return True
    if path.suffix in SKIP_SUFFIXES:
        return True
    if _is_output_file_to_skip(path):
        return True
    return any(_is_relative_to(path, prefix) for prefix in SKIP_PREFIXES)


def package_paths() -> list[Path]:
    package_files = [
        path
        for path in sorted(ROOT.rglob("*"))
        if path != ARCHIVE and path.is_file() and not should_skip(path)
    ]
    package_files.extend(path for path in EXTRA_PACKAGE_FILES if path.is_file())
    for directory in EXTRA_PACKAGE_DIRS:
        if directory.is_dir():
            package_files.extend(
                path
                for path in sorted(directory.rglob("*"))
                if path.is_file() and path.suffix.lower() in EXTRA_PACKAGE_SUFFIXES
            )
    return sorted(package_files)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a safe submission zip.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the files that would be packaged without writing the archive.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = package_paths()

    if args.dry_run:
        for path in paths:
            print(path.relative_to(ROOT.parent))
        print(f"Would package {len(paths)} files")
        return 0

    if ARCHIVE.exists():
        ARCHIVE.unlink()

    with zipfile.ZipFile(ARCHIVE, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in paths:
            zf.write(path, path.relative_to(ROOT.parent))

    print(f"Wrote {ARCHIVE} with {len(paths)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
