#!/usr/bin/env python3
"""Build standalone AliMeeting/M2MeT experiment configs for TRS validation.

This script does not modify the project source code. It scans an AliMeeting-like
dataset tree, extracts references from Praat TextGrid files when available, and
writes one config per requested ASR model under ``server_experiments/``.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import re
import wave


DEFAULT_DATASET_ROOT = "/root/autodl-tmp/moved/datasets/AliMeeting"
DEFAULT_OUTPUT_DIR = "server_experiments"
DEFAULT_EXPERIMENT_OUTPUT_ROOT = "sever_outputs"
DEFAULT_MAX_HOURS = 2.0
DEFAULT_ASR_MODELS = [
    "funasr",
]
DEFAULT_PIPELINES = [
    "direct_asr",
    "diarization_asr",
    "diarization_turn_asr",
    "separation_asr",
]

TEXTGRID_SUFFIXES = {".textgrid", ".TextGrid", ".TEXTGRID"}
AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus"}
SKIP_TEXT = {
    "",
    "<sil>",
    "<noise>",
    "<unk>",
    "sil",
    "noise",
    "unknown",
}


@dataclass(frozen=True)
class Turn:
    speaker: str
    start: float
    end: float
    text: str


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    dataset_root = Path(args.dataset_root).resolve()
    output_dir = (project_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    textgrid_index = index_textgrids(dataset_root)
    if args.scan_only:
        print_scan_report(dataset_root, textgrid_index)
        return 0

    samples = build_samples(
        dataset_root=dataset_root,
        textgrid_index=textgrid_index,
        max_hours=args.max_hours,
        allow_unannotated=args.allow_unannotated,
    )
    if not samples:
        raise SystemExit(
            "No usable samples found. Run with --scan-only to inspect audio and "
            "TextGrid counts, or use --allow-unannotated for a smoke run without "
            "TRS references."
        )

    written_paths = []
    if args.single_config:
        output_name = args.single_config_output or args.single_config
        path = output_dir / f"{args.single_config}.json"
        write_config(
            path,
            description=(
                f"{args.experiment_label} comparing "
                f"{', '.join(args.asr_model)} with TRS Speaker evaluation."
            ),
            asr_model=args.asr_model[0],
            asr_models=args.asr_model,
            output_name=output_name,
            experiment_output_root=args.experiment_output_root,
            pipelines=args.pipeline,
            samples=samples,
        )
        written_paths.append(path)
    else:
        for asr_model in args.asr_model:
            slug = model_slug(asr_model)
            output_name = f"{args.output_prefix}_{slug}"
            path = output_dir / f"{output_name}.json"
            write_config(
                path,
                description=(
                    f"{args.experiment_label} with {asr_model} timestamp output "
                    "and TRS Speaker evaluation."
                ),
                asr_model=asr_model,
                asr_models=[asr_model],
                output_name=output_name,
                experiment_output_root=args.experiment_output_root,
                pipelines=args.pipeline,
                samples=samples,
            )
            written_paths.append(path)

    total_hours = sum(float(sample["_duration_seconds"]) for sample in samples) / 3600
    print(f"Wrote {len(samples)} samples ({total_hours:.2f}h) to {output_dir}")
    for path in written_paths:
        print(f"- {path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--experiment-output-root", default=DEFAULT_EXPERIMENT_OUTPUT_ROOT)
    parser.add_argument("--max-hours", type=float, default=DEFAULT_MAX_HOURS)
    parser.add_argument(
        "--output-prefix",
        default="alimeeting",
        help="Prefix for generated per-model config names and output directories.",
    )
    parser.add_argument(
        "--single-config",
        default="",
        help="Write one config with all --asr-model values instead of one config per model.",
    )
    parser.add_argument(
        "--single-config-output",
        default="",
        help="Output directory name for --single-config. Defaults to the config stem.",
    )
    parser.add_argument(
        "--experiment-label",
        default="AliMeeting/M2MeT ASR benchmark",
        help="Human-readable description prefix written into generated configs.",
    )
    parser.add_argument(
        "--asr-model",
        action="append",
        default=[],
        help=(
            "ASR model to benchmark. Repeat this option for multiple models. "
            "Defaults to funasr."
        ),
    )
    parser.add_argument(
        "--pipeline",
        action="append",
        default=[],
        choices=DEFAULT_PIPELINES + ["llm_rag_refine"],
        help=(
            "Pipeline to run. Repeat this option for multiple pipelines. "
            "Defaults to the full non-LLM benchmark set."
        ),
    )
    parser.add_argument(
        "--scan-only",
        action="store_true",
        help="Print dataset scan diagnostics and exit without writing configs.",
    )
    parser.add_argument(
        "--allow-unannotated",
        action="store_true",
        help="Include WAV files without TextGrid references. TRS cannot be computed for them.",
    )
    args = parser.parse_args()
    if not args.asr_model:
        args.asr_model = list(DEFAULT_ASR_MODELS)
    if not args.pipeline:
        args.pipeline = list(DEFAULT_PIPELINES)
    return args


def model_slug(model_name: str) -> str:
    slug = model_name.lower().replace(":", "_").replace("/", "_")
    slug = re.sub(r"[^a-z0-9_]+", "_", slug)
    return re.sub(r"_+", "_", slug).strip("_")


def index_textgrids(dataset_root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in dataset_root.rglob("*"):
        if path.is_file() and path.suffix in TEXTGRID_SUFFIXES:
            index.setdefault(path.stem, path)
    return index


def print_scan_report(dataset_root: Path, textgrid_index: dict[str, Path]) -> None:
    files = [path for path in dataset_root.rglob("*") if path.is_file()]
    suffix_counts = Counter(path.suffix.lower() or "<no_suffix>" for path in files)
    audio_files = [
        path for path in files if path.suffix.lower() in AUDIO_SUFFIXES
    ]
    matched_audio = [
        path for path in audio_files if matching_textgrid_path(path, textgrid_index)
    ]
    nonempty_textgrids = 0
    for textgrid_path in textgrid_index.values():
        if parse_textgrid(textgrid_path):
            nonempty_textgrids += 1

    print(f"Dataset root: {dataset_root}")
    print(f"Total files: {len(files)}")
    print(f"Audio files: {len(audio_files)}")
    print(f"TextGrid files: {len(textgrid_index)}")
    print(f"Audio/TextGrid stem matches: {len(matched_audio)}")
    print(
        "TextGrid files with non-empty parsed turns: "
        f"{nonempty_textgrids}/{len(textgrid_index)}"
    )
    print("Top file suffixes:")
    for suffix, count in suffix_counts.most_common(20):
        print(f"  {suffix}: {count}")
    print("Example audio files:")
    for path in audio_files[:10]:
        print(f"  {path}")
    print("Example TextGrid files:")
    for path in list(textgrid_index.values())[:10]:
        print(f"  {path}")


def build_samples(
    dataset_root: Path,
    textgrid_index: dict[str, Path],
    max_hours: float,
    allow_unannotated: bool,
) -> list[dict[str, object]]:
    audio_paths = sorted(
        path for path in dataset_root.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES
    )
    samples: list[dict[str, object]] = []
    max_seconds = max_hours * 3600
    used_seconds = 0.0

    for audio_path in audio_paths:
        duration = audio_duration_seconds(audio_path)
        if duration <= 0:
            continue

        textgrid_path = matching_textgrid_path(audio_path, textgrid_index)
        turns = parse_textgrid(textgrid_path) if textgrid_path else []
        if not turns and not allow_unannotated:
            continue

        if samples and used_seconds + duration > max_seconds:
            break

        overlap_ratio = compute_overlap_ratio(turns)
        sample = {
            "id": safe_id(audio_path, dataset_root),
            "audio_path": str(audio_path),
            "overlap_level": overlap_level(overlap_ratio),
            "overlap_ratio": overlap_ratio,
            "speakers": max(1, len({turn.speaker for turn in turns}) or 2),
            "_duration_seconds": round(duration, 3),
        }
        reference = flat_reference(turns)
        if reference:
            sample["reference"] = reference
        reference_speakers = speaker_references(turns)
        if reference_speakers:
            sample["reference_speakers"] = reference_speakers
            sample["reference_mode"] = "speaker_block"

        samples.append(sample)
        used_seconds += duration

    return samples


def audio_duration_seconds(path: Path) -> float:
    if path.suffix.lower() == ".wav":
        try:
            with wave.open(str(path), "rb") as handle:
                frames = handle.getnframes()
                rate = handle.getframerate()
                return frames / rate if rate else 0.0
        except (wave.Error, OSError):
            pass

    try:
        import soundfile as sf

        with sf.SoundFile(str(path)) as handle:
            return len(handle) / handle.samplerate if handle.samplerate else 0.0
    except Exception:
        return 0.0


def matching_textgrid_path(audio_path: Path, textgrid_index: dict[str, Path]) -> Path | None:
    """Find the AliMeeting TextGrid for an audio file.

    Near-field audio usually has an exact matching TextGrid stem, such as
    ``R8001_M8004_N_SPK8013``. Far-field audio often has a microphone suffix,
    such as ``R8001_M8004_MS801.wav``, while the TextGrid stem is just
    ``R8001_M8004``.
    """

    exact = textgrid_index.get(audio_path.stem)
    if exact is not None:
        return exact

    far_session_stem = re.sub(r"_MS\d+$", "", audio_path.stem)
    if far_session_stem != audio_path.stem:
        return textgrid_index.get(far_session_stem)
    return None


def parse_textgrid(path: Path | None) -> list[Turn]:
    if path is None or not path.exists():
        return []
    text = path.read_text(encoding="utf-8", errors="ignore")
    turns: list[Turn] = []
    item_blocks = re.findall(
        r"item \[\d+\]:(.*?)(?=\n\s*item \[\d+\]:|\Z)",
        text,
        flags=re.S,
    )
    for block in item_blocks:
        name_match = re.search(r'name\s*=\s*"([^"]*)"', block)
        speaker = clean_speaker(name_match.group(1) if name_match else "speaker")
        interval_blocks = re.findall(
            r"intervals \[\d+\]:(.*?)(?=\n\s*intervals \[\d+\]:|\Z)",
            block,
            flags=re.S,
        )
        for interval in interval_blocks:
            start = field_float(interval, "xmin")
            end = field_float(interval, "xmax")
            raw_text = field_text(interval)
            utterance = clean_text(raw_text)
            if start is None or end is None or end <= start or not utterance:
                continue
            turns.append(Turn(speaker=speaker, start=start, end=end, text=utterance))
    return sorted(turns, key=lambda turn: (turn.start, turn.end, turn.speaker))


def field_float(block: str, name: str) -> float | None:
    match = re.search(rf"{name}\s*=\s*([0-9.]+)", block)
    return float(match.group(1)) if match else None


def field_text(block: str) -> str:
    match = re.search(r'text\s*=\s*"([^"]*)"', block)
    return match.group(1) if match else ""


def clean_speaker(value: str) -> str:
    normalized = re.sub(r"\s+", "_", value.strip())
    return normalized or "speaker"


def clean_text(value: str) -> str:
    text = value.replace("\ufeff", "").strip()
    text = re.sub(r"\s+", "", text)
    return "" if text.lower() in SKIP_TEXT else text


def compute_overlap_ratio(turns: list[Turn]) -> float | None:
    events: list[tuple[float, int]] = []
    for turn in turns:
        events.append((turn.start, 1))
        events.append((turn.end, -1))
    if not events:
        return None
    events.sort(key=lambda item: (item[0], item[1]))

    active = 0
    previous: float | None = None
    speech = 0.0
    overlap = 0.0
    for timestamp, delta in events:
        if previous is not None and timestamp > previous:
            duration = timestamp - previous
            if active > 0:
                speech += duration
            if active >= 2:
                overlap += duration
        active += delta
        previous = timestamp
    return round(overlap / speech, 4) if speech > 0 else None


def overlap_level(ratio: float | None) -> str:
    if ratio is None:
        return "unknown"
    if ratio < 0.05:
        return "low"
    if ratio < 0.20:
        return "medium"
    return "high"


def flat_reference(turns: list[Turn]) -> str:
    return " ".join(turn.text for turn in turns if turn.text).strip()


def speaker_references(turns: list[Turn]) -> list[dict[str, str]]:
    grouped: dict[str, list[str]] = {}
    for turn in turns:
        grouped.setdefault(turn.speaker, []).append(turn.text)
    return [
        {"speaker": speaker, "text": " ".join(parts).strip()}
        for speaker, parts in sorted(grouped.items())
        if " ".join(parts).strip()
    ]


def safe_id(path: Path, dataset_root: Path) -> str:
    relative = path.relative_to(dataset_root).with_suffix("")
    return re.sub(r"[^A-Za-z0-9_]+", "_", "_".join(relative.parts)).strip("_")


def write_config(
    path: Path,
    description: str,
    asr_model: str,
    asr_models: list[str],
    output_name: str,
    experiment_output_root: str,
    pipelines: list[str],
    samples: list[dict[str, object]],
) -> None:
    clean_samples = []
    for sample in samples:
        clean_sample = dict(sample)
        clean_sample.pop("_duration_seconds", None)
        clean_samples.append(clean_sample)

    config = {
        "project_name": "overlap_asr_llm",
        "description": description,
        "language": "zh",
        "asr_prompt": (
            "以下是中文多人会议录音。请使用简体中文逐字转写，"
            "不要添加音频中没有出现的内容。"
        ),
        "output_dir": str(Path(experiment_output_root).expanduser() / output_name),
        "models": {
            "asr": asr_model,
            "diarization": "pyannote:pyannote/speaker-diarization-community-1",
            "separation": "clearvoice:MossFormer2_SS_16K",
            "llm": "mock",
        },
        "asr_models": asr_models,
        "pipelines": pipelines,
        "rag_context": [],
        "samples": clean_samples,
    }
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
