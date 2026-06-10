"""Experiment pipeline orchestration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import time

from .config import ExperimentConfig, Sample
from .metrics import cer, wer
from .providers import make_asr, make_diarizer, make_llm_refiner, make_separator


@dataclass
class PipelineResult:
    sample_id: str
    audio_path: str
    overlap_level: str
    pipeline: str
    model: str
    text: str
    speaker_labels: str
    runtime_seconds: float
    cer: float | None
    wer: float | None
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _score(reference: str | None, text: str) -> tuple[float | None, float | None]:
    if not reference:
        return None, None
    return cer(reference, text), wer(reference, text)


def _labels_from_segments(segments: list[dict[str, object]]) -> str:
    labels = [str(segment.get("speaker", "UNKNOWN")) for segment in segments]
    return ",".join(dict.fromkeys(labels))


def _error_result(
    sample: Sample,
    pipeline: str,
    model: str,
    started: float,
    error: Exception,
) -> PipelineResult:
    return PipelineResult(
        sample_id=sample.id,
        audio_path=str(sample.audio_path),
        overlap_level=sample.overlap_level,
        pipeline=pipeline,
        model=model,
        text="",
        speaker_labels="",
        runtime_seconds=round(time.perf_counter() - started, 4),
        cer=None,
        wer=None,
        error=f"{type(error).__name__}: {error}",
    )


def run_direct_asr(config: ExperimentConfig, sample: Sample) -> PipelineResult:
    started = time.perf_counter()
    model_name = config.models.get("asr", "mock")
    try:
        asr = make_asr(model_name)
        transcript = asr.transcribe(sample.audio_path, config.language)
        cer_value, wer_value = _score(sample.reference, transcript.text)
        return PipelineResult(
            sample_id=sample.id,
            audio_path=str(sample.audio_path),
            overlap_level=sample.overlap_level,
            pipeline="direct_asr",
            model=asr.name,
            text=transcript.text,
            speaker_labels=_labels_from_segments(transcript.segments),
            runtime_seconds=round(time.perf_counter() - started, 4),
            cer=cer_value,
            wer=wer_value,
        )
    except Exception as exc:
        return _error_result(sample, "direct_asr", model_name, started, exc)


def run_diarization_asr(config: ExperimentConfig, sample: Sample) -> PipelineResult:
    started = time.perf_counter()
    asr_name = config.models.get("asr", "mock")
    diarizer_name = config.models.get("diarization", "mock")
    model_name = f"{asr_name}+{diarizer_name}"
    try:
        asr = make_asr(asr_name)
        diarizer = make_diarizer(diarizer_name)
        transcript = asr.transcribe(sample.audio_path, config.language)
        labeled = diarizer.label(transcript, sample.speakers, audio_path=sample.audio_path)
        cer_value, wer_value = _score(sample.reference, labeled.text)
        return PipelineResult(
            sample_id=sample.id,
            audio_path=str(sample.audio_path),
            overlap_level=sample.overlap_level,
            pipeline="diarization_asr",
            model=f"{asr.name}+{diarizer.name}",
            text=labeled.text,
            speaker_labels=_labels_from_segments(labeled.segments),
            runtime_seconds=round(time.perf_counter() - started, 4),
            cer=cer_value,
            wer=wer_value,
        )
    except Exception as exc:
        return _error_result(sample, "diarization_asr", model_name, started, exc)


def run_separation_asr(config: ExperimentConfig, sample: Sample) -> PipelineResult:
    started = time.perf_counter()
    asr_name = config.models.get("asr", "mock")
    separator_name = config.models.get("separation", "mock")
    model_name = f"{separator_name}+{asr_name}"
    try:
        asr = make_asr(asr_name)
        separator = make_separator(separator_name)
        separated_dir = config.output_dir / "separated_audio" / sample.id
        speaker_paths = separator.separate(sample.audio_path, separated_dir, sample.speakers)
        parts = []
        segments = []
        for index, speaker_path in enumerate(speaker_paths, start=1):
            transcript = asr.transcribe(speaker_path, config.language)
            speaker = f"SPEAKER{index}"
            parts.append(f"[{speaker}] {transcript.text}")
            segments.extend({**segment, "speaker": speaker} for segment in transcript.segments)
        text = " ".join(parts)
        cer_value, wer_value = _score(sample.reference, text)
        return PipelineResult(
            sample_id=sample.id,
            audio_path=str(sample.audio_path),
            overlap_level=sample.overlap_level,
            pipeline="separation_asr",
            model=f"{separator.name}+{asr.name}",
            text=text,
            speaker_labels=_labels_from_segments(segments),
            runtime_seconds=round(time.perf_counter() - started, 4),
            cer=cer_value,
            wer=wer_value,
        )
    except Exception as exc:
        return _error_result(sample, "separation_asr", model_name, started, exc)


def run_llm_rag_refine(
    config: ExperimentConfig,
    sample: Sample,
    source_results: list[PipelineResult],
) -> PipelineResult:
    started = time.perf_counter()
    llm_name = config.models.get("llm", "mock")
    try:
        refiner = make_llm_refiner(llm_name)
        source_text = "\n".join(
            f"{result.pipeline}: {result.text}"
            for result in source_results
            if result.sample_id == sample.id and not result.error
        )
        refined = refiner.refine(source_text, config.rag_context)
        cer_value, wer_value = _score(sample.reference, refined)
        return PipelineResult(
            sample_id=sample.id,
            audio_path=str(sample.audio_path),
            overlap_level=sample.overlap_level,
            pipeline="llm_rag_refine",
            model=refiner.name,
            text=refined,
            speaker_labels="LLM_REFINED",
            runtime_seconds=round(time.perf_counter() - started, 4),
            cer=cer_value,
            wer=wer_value,
        )
    except Exception as exc:
        return _error_result(sample, "llm_rag_refine", llm_name, started, exc)


def run_all(config: ExperimentConfig) -> list[PipelineResult]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    results: list[PipelineResult] = []
    for sample in config.samples:
        if "direct_asr" in config.pipelines:
            results.append(run_direct_asr(config, sample))
        if "diarization_asr" in config.pipelines:
            results.append(run_diarization_asr(config, sample))
        if "separation_asr" in config.pipelines:
            results.append(run_separation_asr(config, sample))
        if "llm_rag_refine" in config.pipelines:
            results.append(run_llm_rag_refine(config, sample, results))
    return results
