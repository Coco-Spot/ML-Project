"""Experiment pipeline orchestration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import json
from pathlib import Path
import re
import tempfile
import time

from .config import ExperimentConfig, LLMRAGSource, Sample
from .metrics import cer, speaker_block_score, wer
from .providers import make_asr, make_diarizer, make_llm_refiner, make_separator
from .rag import infer_domain_tags, retrieve_rag_context, tags_for_sample


_SUBTITLE_PREFIX_RE = re.compile(
    r"^\s*\d{2}:\d{2}:\d{2}(?:[.,]\d+)?\s*-->\s*"
    r"\d{2}:\d{2}:\d{2}(?:[.,]\d+)?\s*"
)
_SPEAKER_PREFIX_RE = re.compile(r"^\s*\[[^\]]+\]\s*")
_MIN_TURN_ASR_SECONDS = 0.2


class ProviderCache:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self._asr = None
        self._diarizer = None
        self._separator = None
        self._refiner = None

    def asr(self):
        if self._asr is None:
            self._asr = make_asr(self.config.models.get("asr", "mock"))
        return self._asr

    def diarizer(self):
        if self._diarizer is None:
            self._diarizer = make_diarizer(self.config.models.get("diarization", "mock"))
        return self._diarizer

    def separator(self):
        if self._separator is None:
            self._separator = make_separator(self.config.models.get("separation", "mock"))
        return self._separator

    def refiner(self):
        if self._refiner is None:
            self._refiner = make_llm_refiner(self.config.models.get("llm", "mock"))
        return self._refiner


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
    text_cer: float | None = None
    text_wer: float | None = None
    flat_cer: float | None = None
    flat_wer: float | None = None
    timeline_cer: float | None = None
    timeline_wer: float | None = None
    speaker_block_cer: float | None = None
    speaker_block_wer: float | None = None
    best_speaker_mapping: str = ""
    score_basis: str = "flat"
    error: str | None = None
    segments: list[dict[str, object]] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _score(reference: str | None, text: str) -> tuple[float | None, float | None]:
    if not reference:
        return None, None
    return cer(reference, text), wer(reference, text)


def _refined_text_for_scoring(text: str) -> str:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return _spoken_text_from_subtitle(text)
    if isinstance(data, dict):
        refined = data.get("refined_text")
        if isinstance(refined, str) and refined.strip():
            return _spoken_text_from_subtitle(refined)
    return _spoken_text_from_subtitle(text)


def _spoken_text_from_subtitle(text: str) -> str:
    spoken_lines = []
    for line in text.splitlines():
        line = _SUBTITLE_PREFIX_RE.sub("", line.strip())
        line = _SPEAKER_PREFIX_RE.sub("", line).strip()
        if line:
            spoken_lines.append(line)
    return " ".join(spoken_lines).strip()


def _labels_from_segments(segments: list[dict[str, object]]) -> str:
    labels = [str(segment.get("speaker", "UNKNOWN")) for segment in segments]
    return ",".join(dict.fromkeys(labels))


def _format_timestamp(seconds: float) -> str:
    seconds = max(seconds, 0.0)
    minutes, second_value = divmod(seconds, 60)
    hours, minute_value = divmod(int(minutes), 60)
    return f"{hours:02d}:{minute_value:02d}:{second_value:06.3f}"


def _subtitle_text(segments: list[dict[str, object]]) -> str:
    lines = []
    for segment in segments:
        start = _format_timestamp(float(segment.get("start", 0.0)))
        end = _format_timestamp(float(segment.get("end", 0.0)))
        speaker = str(segment.get("speaker", "UNKNOWN"))
        text = str(segment.get("text", "")).strip()
        lines.append(f"{start} --> {end} [{speaker}] {text}".strip())
    return "\n".join(lines)


def _plain_text_from_segments(segments: list[dict[str, object]]) -> str:
    return " ".join(str(segment.get("text", "")).strip() for segment in segments).strip()


def _timeline_text_from_segments(segments: list[dict[str, object]]) -> str:
    ordered = sorted(
        segments,
        key=lambda segment: (
            float(segment.get("start", 0.0)),
            float(segment.get("end", 0.0)),
            str(segment.get("speaker", "")),
        ),
    )
    return _plain_text_from_segments(ordered)


def _speaker_texts_from_segments(
    segments: list[dict[str, object]],
) -> dict[str, str]:
    speaker_texts: dict[str, list[str]] = {}
    for segment in segments:
        text = str(segment.get("text", "")).strip()
        if not text:
            continue
        speaker = str(segment.get("speaker", "UNKNOWN"))
        speaker_texts.setdefault(speaker, []).append(text)
    return {
        speaker: " ".join(parts).strip()
        for speaker, parts in speaker_texts.items()
        if " ".join(parts).strip()
    }


@dataclass(frozen=True)
class ScoreBundle:
    cer: float | None
    wer: float | None
    text_cer: float | None
    text_wer: float | None
    flat_cer: float | None
    flat_wer: float | None
    timeline_cer: float | None
    timeline_wer: float | None
    speaker_block_cer: float | None
    speaker_block_wer: float | None
    best_speaker_mapping: str
    score_basis: str


def _score_bundle(
    sample: Sample,
    flat_text: str,
    segments: list[dict[str, object]] | None = None,
    prefer_speaker_block: bool = False,
) -> ScoreBundle:
    flat_cer, flat_wer = _score(sample.reference, flat_text)
    timeline_text = _timeline_text_from_segments(segments or []) if segments else flat_text
    timeline_cer, timeline_wer = _score(sample.reference, timeline_text)

    speaker_cer = None
    speaker_wer = None
    best_mapping = ""
    if sample.reference_speakers and segments:
        reference_speakers = {
            item.speaker: item.text for item in sample.reference_speakers
        }
        hypothesis_speakers = _speaker_texts_from_segments(segments)
        if len(hypothesis_speakers) >= 2:
            speaker_score = speaker_block_score(
                reference_speakers,
                hypothesis_speakers,
            )
            if speaker_score is not None:
                speaker_cer = speaker_score.cer
                speaker_wer = speaker_score.wer
                best_mapping = json.dumps(
                    speaker_score.mapping,
                    ensure_ascii=False,
                    sort_keys=True,
                )

    if prefer_speaker_block and speaker_cer is not None and speaker_wer is not None:
        primary_cer = speaker_cer
        primary_wer = speaker_wer
        score_basis = "speaker_block"
    elif timeline_cer is not None and timeline_wer is not None:
        primary_cer = timeline_cer
        primary_wer = timeline_wer
        score_basis = "timeline"
    else:
        primary_cer = flat_cer
        primary_wer = flat_wer
        score_basis = "flat"

    return ScoreBundle(
        cer=primary_cer,
        wer=primary_wer,
        text_cer=timeline_cer,
        text_wer=timeline_wer,
        flat_cer=flat_cer,
        flat_wer=flat_wer,
        timeline_cer=timeline_cer,
        timeline_wer=timeline_wer,
        speaker_block_cer=speaker_cer,
        speaker_block_wer=speaker_wer,
        best_speaker_mapping=best_mapping,
        score_basis=score_basis,
    )


def _result_scores(score: ScoreBundle) -> dict[str, object]:
    return {
        "cer": score.cer,
        "wer": score.wer,
        "text_cer": score.text_cer,
        "text_wer": score.text_wer,
        "flat_cer": score.flat_cer,
        "flat_wer": score.flat_wer,
        "timeline_cer": score.timeline_cer,
        "timeline_wer": score.timeline_wer,
        "speaker_block_cer": score.speaker_block_cer,
        "speaker_block_wer": score.speaker_block_wer,
        "best_speaker_mapping": score.best_speaker_mapping,
        "score_basis": score.score_basis,
    }


def _write_audio_excerpt(
    audio_path: Path,
    start: float,
    end: float,
    output_path: Path,
) -> None:
    import soundfile as sf

    with sf.SoundFile(str(audio_path)) as source:
        sample_rate = source.samplerate
        total_frames = len(source)
        start_frame = max(0, min(total_frames, int(start * sample_rate)))
        end_frame = max(0, min(total_frames, int(end * sample_rate)))
        if end_frame <= start_frame:
            end_frame = min(total_frames, start_frame + max(1, int(0.1 * sample_rate)))
        source.seek(start_frame)
        audio = source.read(end_frame - start_frame, dtype="float32", always_2d=True)
    sf.write(str(output_path), audio, sample_rate)


def _transcribe_speaker_turns(
    asr,
    audio_path: Path,
    speaker_turns: list[dict[str, object]],
    language: str,
    prompt: str | None = None,
) -> list[dict[str, object]]:
    ordered_turns = sorted(
        speaker_turns,
        key=lambda segment: float(segment.get("start", 0.0)),
    )
    if getattr(asr, "name", "") == "mock_asr":
        transcript = asr.transcribe(audio_path, language, prompt=prompt)
        base_text = transcript.text.strip()
        return [
            {
                "start": float(turn.get("start", 0.0)),
                "end": float(turn.get("end", turn.get("start", 0.0))),
                "speaker": str(turn.get("speaker", "UNKNOWN")),
                "text": str(turn.get("text", "")).strip() or f"{base_text} turn {index}",
            }
            for index, turn in enumerate(ordered_turns, start=1)
        ]

    segments: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="overlap_asr_llm_diarization_") as tmpdir:
        tmp_path = Path(tmpdir)
        for index, turn in enumerate(ordered_turns, start=1):
            start = float(turn.get("start", 0.0))
            end = float(turn.get("end", start))
            if end - start < _MIN_TURN_ASR_SECONDS:
                continue
            excerpt_path = tmp_path / f"turn_{index:04d}.wav"
            _write_audio_excerpt(audio_path, start, end, excerpt_path)
            try:
                transcript = asr.transcribe(excerpt_path, language, prompt=prompt)
            except Exception:
                continue
            text = transcript.text.strip()
            if not text:
                text = " ".join(
                    str(segment.get("text", "")).strip()
                    for segment in transcript.segments
                    if str(segment.get("text", "")).strip()
                )
            segments.append(
                {
                    "start": start,
                    "end": end,
                    "speaker": str(turn.get("speaker", "UNKNOWN")),
                    "text": text,
                }
            )
    return segments


def _best_speaker_for_segment(
    segment: dict[str, object],
    speaker_turns: list[dict[str, object]],
) -> str:
    if not speaker_turns:
        return "UNKNOWN"

    start = float(segment.get("start", 0.0))
    end = float(segment.get("end", start))
    if end <= start:
        end = start + 0.01

    best_label = "UNKNOWN"
    best_overlap = 0.0
    for turn in speaker_turns:
        turn_start = float(turn.get("start", 0.0))
        turn_end = float(turn.get("end", turn_start))
        overlap = max(0.0, min(end, turn_end) - max(start, turn_start))
        if overlap > best_overlap:
            best_overlap = overlap
            best_label = str(turn.get("speaker", "UNKNOWN"))

    if best_overlap > 0:
        return best_label

    midpoint = (start + end) / 2
    nearest = min(
        speaker_turns,
        key=lambda turn: min(
            abs(midpoint - float(turn.get("start", 0.0))),
            abs(midpoint - float(turn.get("end", turn.get("start", 0.0)))),
        ),
    )
    return str(nearest.get("speaker", "UNKNOWN"))


def _label_transcript_segments(
    transcript_segments: list[dict[str, object]],
    speaker_turns: list[dict[str, object]],
) -> list[dict[str, object]]:
    return [
        {
            **segment,
            "speaker": _best_speaker_for_segment(segment, speaker_turns),
        }
        for segment in transcript_segments
        if str(segment.get("text", "")).strip()
    ]


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
        text_cer=None,
        text_wer=None,
        flat_cer=None,
        flat_wer=None,
        timeline_cer=None,
        timeline_wer=None,
        speaker_block_cer=None,
        speaker_block_wer=None,
        best_speaker_mapping="",
        score_basis="error",
        error=f"{type(error).__name__}: {error}",
        segments=[],
    )


def run_direct_asr(
    config: ExperimentConfig,
    sample: Sample,
    providers: ProviderCache | None = None,
) -> PipelineResult:
    started = time.perf_counter()
    model_name = config.models.get("asr", "mock")
    try:
        asr = providers.asr() if providers else make_asr(model_name)
        started = time.perf_counter()
        transcript = asr.transcribe(
            sample.audio_path,
            config.language,
            prompt=config.asr_prompt,
        )
        score = _score_bundle(
            sample,
            transcript.text,
            transcript.segments,
            prefer_speaker_block=False,
        )
        return PipelineResult(
            sample_id=sample.id,
            audio_path=str(sample.audio_path),
            overlap_level=sample.overlap_level,
            pipeline="direct_asr",
            model=model_name,
            text=transcript.text,
            speaker_labels=_labels_from_segments(transcript.segments),
            runtime_seconds=round(time.perf_counter() - started, 4),
            **_result_scores(score),
            segments=transcript.segments,
        )
    except Exception as exc:
        return _error_result(sample, "direct_asr", model_name, started, exc)


def run_diarization_asr(
    config: ExperimentConfig,
    sample: Sample,
    providers: ProviderCache | None = None,
) -> PipelineResult:
    return _run_diarization_asr(
        config,
        sample,
        pipeline="diarization_asr",
        mode="full_asr_align",
        providers=providers,
    )


def run_diarization_turn_asr(
    config: ExperimentConfig,
    sample: Sample,
    providers: ProviderCache | None = None,
) -> PipelineResult:
    return _run_diarization_asr(
        config,
        sample,
        pipeline="diarization_turn_asr",
        mode="turn_asr",
        providers=providers,
    )


def _run_diarization_asr(
    config: ExperimentConfig,
    sample: Sample,
    pipeline: str,
    mode: str,
    providers: ProviderCache | None = None,
) -> PipelineResult:
    started = time.perf_counter()
    asr_name = config.models.get("asr", "mock")
    diarizer_name = config.models.get("diarization", "mock")
    model_name = f"{asr_name}+{diarizer_name}"
    try:
        diarizer = providers.diarizer() if providers else make_diarizer(diarizer_name)
        started = time.perf_counter()
        speaker_turns = diarizer.diarize(sample.audio_path, sample.speakers)
        if hasattr(diarizer, "release_gpu"):
            diarizer.release_gpu()
        asr = providers.asr() if providers else make_asr(asr_name)
        if mode == "turn_asr":
            segments = _transcribe_speaker_turns(
                asr,
                sample.audio_path,
                speaker_turns,
                config.language,
                prompt=config.asr_prompt,
            )
        else:
            transcript = asr.transcribe(
                sample.audio_path,
                config.language,
                prompt=config.asr_prompt,
            )
            segments = _label_transcript_segments(transcript.segments, speaker_turns)
        if hasattr(asr, "release_gpu"):
            asr.release_gpu()
        subtitle_text = _subtitle_text(segments)
        plain_text = _plain_text_from_segments(segments)
        score = _score_bundle(
            sample,
            plain_text,
            segments,
            prefer_speaker_block=True,
        )
        return PipelineResult(
            sample_id=sample.id,
            audio_path=str(sample.audio_path),
            overlap_level=sample.overlap_level,
            pipeline=pipeline,
            model=f"{asr_name}+{diarizer_name}+{mode}",
            text=subtitle_text,
            speaker_labels=_labels_from_segments(segments),
            runtime_seconds=round(time.perf_counter() - started, 4),
            **_result_scores(score),
            segments=segments,
        )
    except Exception as exc:
        return _error_result(sample, pipeline, model_name, started, exc)


def run_separation_asr(
    config: ExperimentConfig,
    sample: Sample,
    providers: ProviderCache | None = None,
) -> PipelineResult:
    started = time.perf_counter()
    asr_name = config.models.get("asr", "mock")
    separator_name = config.models.get("separation", "mock")
    model_name = f"{separator_name}+{asr_name}"
    try:
        asr = providers.asr() if providers else make_asr(asr_name)
        separator = providers.separator() if providers else make_separator(separator_name)
        started = time.perf_counter()
        separated_dir = config.output_dir / "separated_audio" / sample.id
        speaker_paths = separator.separate(sample.audio_path, separated_dir, sample.speakers)
        parts = []
        segments = []
        for index, speaker_path in enumerate(speaker_paths, start=1):
            transcript = asr.transcribe(
                speaker_path,
                config.language,
                prompt=config.asr_prompt,
            )
            speaker = f"SPEAKER{index}"
            parts.append(f"[{speaker}] {transcript.text}")
            transcript_segments = transcript.segments or [
                {
                    "start": 0.0,
                    "end": 0.0,
                    "text": transcript.text,
                    "speaker": "UNKNOWN",
                }
            ]
            segments.extend(
                {
                    **segment,
                    "speaker": speaker,
                    "source_index": index,
                    "separated_audio_path": str(speaker_path),
                }
                for segment in transcript_segments
            )
        text = " ".join(parts)
        plain_text = _plain_text_from_segments(segments)
        score = _score_bundle(
            sample,
            plain_text,
            segments,
            prefer_speaker_block=True,
        )
        return PipelineResult(
            sample_id=sample.id,
            audio_path=str(sample.audio_path),
            overlap_level=sample.overlap_level,
            pipeline="separation_asr",
            model=f"{separator_name}+{asr_name}",
            text=text,
            speaker_labels=_labels_from_segments(segments),
            runtime_seconds=round(time.perf_counter() - started, 4),
            **_result_scores(score),
            segments=segments,
        )
    except Exception as exc:
        return _error_result(sample, "separation_asr", model_name, started, exc)


def run_llm_rag_refine(
    config: ExperimentConfig,
    sample: Sample,
    source_results: list[PipelineResult],
    providers: ProviderCache | None = None,
) -> PipelineResult:
    started = time.perf_counter()
    llm_name = config.models.get("llm", "mock")
    try:
        refiner = providers.refiner() if providers else make_llm_refiner(llm_name)
        started = time.perf_counter()
        source_text = "\n".join(
            f"{result.pipeline}: {result.text}"
            for result in source_results
            if result.sample_id == sample.id and not result.error
        )
        rag_tags = tags_for_sample(sample.overlap_level, "llm_rag_refine")
        rag_tags.extend(infer_domain_tags("\n".join((config.asr_prompt or "", source_text))))
        rag_context = retrieve_rag_context(rag_tags, config.rag_context)
        refined = refiner.refine(source_text, rag_context)
        score = _score_bundle(sample, _refined_text_for_scoring(refined))
        return PipelineResult(
            sample_id=sample.id,
            audio_path=str(sample.audio_path),
            overlap_level=sample.overlap_level,
            pipeline="llm_rag_refine",
            model=refiner.name,
            text=refined,
            speaker_labels="LLM_REFINED",
            runtime_seconds=round(time.perf_counter() - started, 4),
            **_result_scores(score),
            segments=[],
        )
    except Exception as exc:
        return _error_result(sample, "llm_rag_refine", llm_name, started, exc)


def run_llm_rag_refine_from_source(
    config: ExperimentConfig,
    sample: Sample,
    source: LLMRAGSource,
) -> PipelineResult:
    started = time.perf_counter()
    source_config = replace(
        config,
        models={**config.models, **source.models},
        pipelines=[source.pipeline],
        llm_rag_sources=(),
    )
    source_result = _run_source_pipeline(source_config, sample, source.pipeline)
    if source_result.error:
        return _error_result(
            sample,
            "llm_rag_refine",
            f"{config.models.get('llm', 'mock')}<-{source.label}:{source_result.model}",
            started,
            RuntimeError(f"source {source.label}: {source_result.error}"),
        )

    llm_result = run_llm_rag_refine(config, sample, [source_result])

    llm_result.model = f"{llm_result.model}<-{source.label}:{source_result.model}"
    llm_result.runtime_seconds = round(
        source_result.runtime_seconds + llm_result.runtime_seconds,
        4,
    )
    llm_result.speaker_labels = source_result.speaker_labels
    llm_result.segments = source_result.segments
    if source_result.error and not llm_result.error:
        llm_result.error = f"source {source.label}: {source_result.error}"
    return llm_result


def _run_source_pipeline(
    config: ExperimentConfig,
    sample: Sample,
    pipeline: str,
) -> PipelineResult:
    if pipeline == "direct_asr":
        return run_direct_asr(config, sample)
    if pipeline == "diarization_asr":
        return run_diarization_asr(config, sample)
    if pipeline == "diarization_turn_asr":
        return run_diarization_turn_asr(config, sample)
    if pipeline == "separation_asr":
        return run_separation_asr(config, sample)
    raise ValueError(f"Unsupported llm_rag source pipeline: {pipeline}")


def run_all(config: ExperimentConfig) -> list[PipelineResult]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    results: list[PipelineResult] = []
    asr_models = config.asr_models or [config.models.get("asr", "mock")]
    for asr_model in asr_models:
        model_config = replace(
            config,
            models={**config.models, "asr": asr_model},
            asr_models=[asr_model],
        )
        providers = ProviderCache(model_config)
        for sample in model_config.samples:
            sample_results: list[PipelineResult] = []
            if "direct_asr" in model_config.pipelines:
                result = run_direct_asr(model_config, sample, providers)
                results.append(result)
                sample_results.append(result)
            if "diarization_asr" in model_config.pipelines:
                result = run_diarization_asr(model_config, sample, providers)
                results.append(result)
                sample_results.append(result)
            if "diarization_turn_asr" in model_config.pipelines:
                result = run_diarization_turn_asr(model_config, sample, providers)
                results.append(result)
                sample_results.append(result)
            if "separation_asr" in model_config.pipelines:
                result = run_separation_asr(model_config, sample, providers)
                results.append(result)
                sample_results.append(result)
            if "llm_rag_refine" in model_config.pipelines:
                if model_config.llm_rag_sources:
                    for source in model_config.llm_rag_sources:
                        results.append(run_llm_rag_refine_from_source(model_config, sample, source))
                else:
                    results.append(
                        run_llm_rag_refine(model_config, sample, sample_results, providers)
                    )
    return results
