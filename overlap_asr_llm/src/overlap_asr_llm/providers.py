"""Model provider wrappers.

The project can run in mock mode on a laptop and in real-model mode on a GPU
machine. Optional heavy dependencies are imported lazily.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import shutil


@dataclass
class Transcript:
    text: str
    segments: list[dict[str, object]]


def _model_cache_dir() -> Path:
    cache_dir = Path(
        os.environ.get("OVERLAP_ASR_LLM_CACHE_DIR", "outputs/caches")
    ).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _prepare_huggingface_download_env() -> Path:
    cache_dir = _model_cache_dir()
    os.environ.setdefault("HF_HOME", str(cache_dir / "huggingface"))
    os.environ.setdefault("HF_HUB_CACHE", str(cache_dir / "huggingface" / "hub"))
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    hf_endpoint = os.environ.get("HF_ENDPOINT", "")
    if "hf-mirror.com" in hf_endpoint:
        os.environ["HF_ENDPOINT"] = "https://huggingface.co"
    return cache_dir


def _load_pyannote_pipeline(Pipeline, model_id: str, token: str, cache_dir: Path):
    return Pipeline.from_pretrained(model_id, token=token, cache_dir=cache_dir)


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _configure_torch_tf32(torch) -> None:
    if not _env_flag("OVERLAP_ASR_LLM_ENABLE_TF32"):
        return
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


class MockASR:
    name = "mock_asr"

    def transcribe(
        self,
        audio_path: Path,
        language: str,
        prompt: str | None = None,
    ) -> Transcript:
        del prompt
        stem = audio_path.stem.replace("_", " ")
        text = f"mock transcript for {stem}"
        return Transcript(
            text=text,
            segments=[
                {"start": 0.0, "end": 2.0, "text": text, "speaker": "UNKNOWN"}
            ],
        )


class WhisperASR:
    name = "whisper"

    def __init__(self, model_name: str = "large-v3") -> None:
        import whisper
        import torch

        self.torch = torch
        _configure_torch_tf32(torch)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = whisper.load_model(model_name)

    def transcribe(
        self,
        audio_path: Path,
        language: str,
        prompt: str | None = None,
    ) -> Transcript:
        if self.device.startswith("cuda"):
            self.model.to(self.device)
        result = self.model.transcribe(
            str(audio_path),
            language=language,
            initial_prompt=prompt,
            verbose=False,
            beam_size=1,
            best_of=1,
            temperature=0.0,
            condition_on_previous_text=False,
            fp16=True,
        )
        segments = [
            {
                "start": float(seg.get("start", 0.0)),
                "end": float(seg.get("end", 0.0)),
                "text": str(seg.get("text", "")).strip(),
                "speaker": "UNKNOWN",
            }
            for seg in result.get("segments", [])
        ]
        return Transcript(text=str(result.get("text", "")).strip(), segments=segments)

    def release_gpu(self) -> None:
        if not self.device.startswith("cuda"):
            return
        self.model.to("cpu")
        self.torch.cuda.empty_cache()


class FasterWhisperASR:
    name = "faster-whisper"

    def __init__(self, model_name: str = "large-v3") -> None:
        cache_dir = _prepare_huggingface_download_env()

        from faster_whisper import WhisperModel
        import torch

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device_index = 0 if self.device == "cuda" else None
        self.compute_type = "int8_float16" if self.device == "cuda" else "int8"
        kwargs = {"device_index": self.device_index} if self.device_index is not None else {}
        self.model = WhisperModel(
            model_name,
            device=self.device,
            compute_type=self.compute_type,
            download_root=str(cache_dir / "faster_whisper"),
            **kwargs,
        )

    def transcribe(
        self,
        audio_path: Path,
        language: str,
        prompt: str | None = None,
    ) -> Transcript:
        segments_iter, _ = self.model.transcribe(
            str(audio_path),
            language=language,
            initial_prompt=prompt,
            beam_size=1,
            temperature=0.0,
            condition_on_previous_text=False,
            vad_filter=False,
        )
        segments = [
            {
                "start": float(segment.start),
                "end": float(segment.end),
                "text": segment.text.strip(),
                "speaker": "UNKNOWN",
            }
            for segment in segments_iter
        ]
        text = " ".join(str(segment["text"]) for segment in segments).strip()
        return Transcript(text=text, segments=segments)


class FunASR:
    name = "funasr"

    def __init__(self) -> None:
        cache_dir = Path(
            os.environ.get("MODELSCOPE_CACHE", "outputs/caches/modelscope")
        ).resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MODELSCOPE_CACHE"] = str(cache_dir)

        from funasr import AutoModel
        import torch

        _configure_torch_tf32(torch)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.model = AutoModel(
            model="iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
            vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
            punc_model="iic/punc_ct-transformer_cn-en-common-vocab471067-large",
            device=device,
            disable_update=True,
        )

    def transcribe(
        self,
        audio_path: Path,
        language: str,
        prompt: str | None = None,
    ) -> Transcript:
        del language, prompt
        result = self.model.generate(
            input=str(audio_path),
            batch_size=1,
            return_raw=True,
            timestamp=True,
        )[0]
        text = str(result.get("text", "")).strip()
        raw_segments = (
            result.get("sentences")
            or result.get("sentence_info")
            or self._segments_from_timestamps(result, text)
            or [{"start": 0.0, "end": 0.0, "text": text}]
        )
        segments = [
            {
                "start": self._timestamp_seconds(segment.get("start", 0.0)),
                "end": self._timestamp_seconds(segment.get("end", 0.0)),
                "text": str(segment.get("text", "")).strip(),
                "speaker": str(segment.get("speaker", segment.get("spk", "UNKNOWN"))),
            }
            for segment in raw_segments
        ]
        return Transcript(text=text, segments=segments)

    @staticmethod
    def _timestamp_seconds(value: object) -> float:
        seconds = float(value or 0.0)
        return seconds / 1000.0 if seconds > 100 else seconds

    def _segments_from_timestamps(
        self,
        result: dict[str, object],
        text: str,
    ) -> list[dict[str, object]]:
        timestamps = result.get("timestamp") or result.get("timestamps")
        if not isinstance(timestamps, list) or not timestamps:
            return []

        pairs = [
            item
            for item in timestamps
            if isinstance(item, (list, tuple)) and len(item) >= 2
        ]
        chars = [char for char in text.replace(" ", "") if char.strip()]
        if not pairs or not chars:
            return []
        if len(pairs) != len(chars):
            return [
                {
                    "start": pairs[0][0],
                    "end": pairs[-1][1],
                    "text": text,
                }
            ]

        segments = []
        start = pairs[0][0]
        chunk = []
        for index, (char, pair) in enumerate(zip(chars, pairs)):
            chunk.append(char)
            is_boundary = char in "。！？!?；;，," or len(chunk) >= 30
            if is_boundary or index == len(chars) - 1:
                segments.append(
                    {
                        "start": start,
                        "end": pair[1],
                        "text": "".join(chunk),
                    }
                )
                if index + 1 < len(pairs):
                    start = pairs[index + 1][0]
                chunk = []
        return segments


class MockDiarizer:
    name = "mock_diarizer"

    def diarize(self, audio_path: Path, speakers: int) -> list[dict[str, object]]:
        del audio_path
        speaker_count = max(speakers, 1)
        return [
            {
                "start": float(index * 2),
                "end": float((index + 1) * 2),
                "speaker": f"SPEAKER{index % speaker_count + 1}",
                "text": "",
            }
            for index in range(speaker_count)
        ]

    def label(
        self,
        transcript: Transcript,
        speakers: int,
        audio_path: Path | None = None,
    ) -> Transcript:
        del audio_path
        labeled = []
        for index, segment in enumerate(transcript.segments):
            speaker_id = index % max(speakers, 1) + 1
            labeled.append({**segment, "speaker": f"SPEAKER{speaker_id}"})
        text = " ".join(
            f"[{segment['speaker']}] {segment['text']}" for segment in labeled
        )
        return Transcript(text=text, segments=labeled)


DEFAULT_PYANNOTE_DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"


class PyannoteDiarizer:
    name = "pyannote"

    def __init__(self, model_id: str = DEFAULT_PYANNOTE_DIARIZATION_MODEL) -> None:
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        if not token:
            raise RuntimeError(
                "Pyannote diarization requires HF_TOKEN. "
                "Create a Hugging Face token and accept the pyannote model terms first."
            )

        cache_dir = _prepare_huggingface_download_env()

        from pyannote.audio import Pipeline
        import torch

        self.torch = torch
        _configure_torch_tf32(torch)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.pipeline = _load_pyannote_pipeline(
            Pipeline,
            model_id,
            token,
            cache_dir / "huggingface",
        )
        if self.pipeline is None:
            raise RuntimeError(
                f"Could not load {model_id}. Make sure your HF_TOKEN is valid "
                "and you accepted the model terms on Hugging Face."
            )
        if self.device.startswith("cuda"):
            self.pipeline.to(torch.device(self.device))

    def diarize(self, audio_path: Path, speakers: int) -> list[dict[str, object]]:
        _configure_torch_tf32(self.torch)
        if self.device.startswith("cuda"):
            self.pipeline.to(self.torch.device(self.device))
        kwargs = {"num_speakers": int(speakers)} if int(speakers) > 0 else {}
        diarization_result = self.pipeline(
            self._audio_input(audio_path),
            **kwargs,
        )
        diarization = self._speaker_diarization_annotation(diarization_result)
        return [
            {
                "start": float(turn.start),
                "end": float(turn.end),
                "speaker": str(label),
                "text": "",
            }
            for turn, _, label in diarization.itertracks(yield_label=True)
        ]

    def release_gpu(self) -> None:
        if not self.device.startswith("cuda"):
            return
        self.pipeline.to(self.torch.device("cpu"))
        self.torch.cuda.empty_cache()

    def label(
        self,
        transcript: Transcript,
        speakers: int,
        audio_path: Path | None = None,
    ) -> Transcript:
        if audio_path is None:
            raise ValueError("Pyannote diarization requires the original audio path.")

        speaker_turns = self.diarize(audio_path, speakers)

        labeled = []
        for segment in transcript.segments:
            speaker = self._best_speaker(segment, speaker_turns)
            labeled_segment = {**segment, "speaker": speaker}
            labeled.append(labeled_segment)
            self._attach_text_to_best_turn(labeled_segment, speaker_turns)
        text = " ".join(
            f"[{segment['speaker']}] {segment['text']}" for segment in labeled
        )
        return Transcript(text=text, segments=speaker_turns)

    @staticmethod
    def _speaker_diarization_annotation(diarization_result):
        return getattr(diarization_result, "speaker_diarization", diarization_result)

    def _audio_input(self, audio_path: Path) -> dict[str, object]:
        try:
            waveform, sample_rate = self._load_waveform(audio_path)
        except Exception:
            return {"audio": str(audio_path), "uri": audio_path.stem}
        return {
            "waveform": waveform,
            "sample_rate": sample_rate,
            "uri": audio_path.stem,
        }

    def _load_waveform(self, audio_path: Path):
        import soundfile as sf

        data, sample_rate = sf.read(
            str(audio_path),
            always_2d=True,
            dtype="float32",
        )
        waveform = self.torch.from_numpy(data.T).contiguous()
        return waveform, int(sample_rate)

    def _best_speaker(
        self,
        segment: dict[str, object],
        speaker_turns: list[dict[str, object]],
    ) -> str:
        if not speaker_turns:
            return "UNKNOWN"

        start = self._to_seconds(float(segment.get("start", 0.0)))
        end = self._to_seconds(float(segment.get("end", start)))
        if end <= start:
            end = start + 0.01

        best_label = "UNKNOWN"
        best_overlap = 0.0
        for turn in speaker_turns:
            turn_start = float(turn["start"])
            turn_end = float(turn["end"])
            overlap = max(0.0, min(end, turn_end) - max(start, turn_start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_label = str(turn["speaker"])

        if best_overlap > 0:
            return best_label

        midpoint = (start + end) / 2
        nearest = min(
            speaker_turns,
            key=lambda turn: min(
                abs(midpoint - float(turn["start"])),
                abs(midpoint - float(turn["end"])),
            ),
        )
        return str(nearest["speaker"])

    def _attach_text_to_best_turn(
        self,
        segment: dict[str, object],
        speaker_turns: list[dict[str, object]],
    ) -> None:
        if not speaker_turns:
            return

        start = self._to_seconds(float(segment.get("start", 0.0)))
        end = self._to_seconds(float(segment.get("end", start)))
        if end <= start:
            return

        best_turn = None
        best_overlap = 0.0
        for turn in speaker_turns:
            overlap = max(
                0.0,
                min(end, float(turn["end"])) - max(start, float(turn["start"])),
            )
            if overlap > best_overlap:
                best_overlap = overlap
                best_turn = turn

        if best_turn is None or best_overlap <= 0:
            return

        text = str(segment.get("text", "")).strip()
        if not text:
            return
        current = str(best_turn.get("text", "")).strip()
        best_turn["text"] = f"{current} {text}".strip()

    @staticmethod
    def _to_seconds(value: float) -> float:
        return value / 1000 if value > 1000 else value


class SpeechBrainDiarizer:
    """Speaker diarization using SpeechBrain embeddings and fixed speaker clustering."""

    name = "speechbrain"

    def __init__(self, model_id: str = "speechbrain/spkrec-ecapa-voxceleb") -> None:
        cache_dir = _prepare_huggingface_download_env()
        try:
            from speechbrain.inference.speaker import SpeakerRecognition
        except ImportError:
            from speechbrain.pretrained import SpeakerRecognition
        import torch

        self.torch = torch
        _configure_torch_tf32(torch)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        try:
            from speechbrain.utils.fetching import LocalStrategy

            self.encoder = SpeakerRecognition.from_hparams(
                source=model_id,
                savedir=str(cache_dir / "speechbrain" / model_id.split("/")[-1]),
                run_opts={"device": self.device},
                local_strategy=LocalStrategy.COPY,
            )
        except TypeError:
            self.encoder = SpeakerRecognition.from_hparams(
                source=model_id,
                savedir=str(cache_dir / "speechbrain" / model_id.split("/")[-1]),
                run_opts={"device": self.device},
            )

    def diarize(self, audio_path: Path, speakers: int) -> list[dict[str, object]]:
        import librosa
        import numpy as np
        import soundfile as sf
        from sklearn.cluster import AgglomerativeClustering

        y, sr = sf.read(str(audio_path), dtype="float32")
        if getattr(y, "ndim", 1) > 1:
            y = y.mean(axis=1)
        if sr != 16000:
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)
            sr = 16000

        intervals = librosa.effects.split(y, top_db=30)
        if len(intervals) == 0:
            intervals = np.array([[0, len(y)]])

        min_frames = int(0.35 * sr)
        usable = [
            (int(start), int(end))
            for start, end in intervals
            if int(end) - int(start) >= min_frames
        ]
        if not usable:
            usable = [(0, len(y))]

        embeddings = []
        turns = []
        for start, end in usable:
            wav = y[start:end]
            with self.torch.no_grad():
                tensor = self.torch.tensor(wav, dtype=self.torch.float32).unsqueeze(0)
                tensor = tensor.to(self.device)
                emb = self.encoder.encode_batch(tensor).squeeze().detach().cpu().numpy()
            embeddings.append(emb)
            turns.append(
                {
                    "start": round(start / sr, 4),
                    "end": round(end / sr, 4),
                    "speaker": "UNKNOWN",
                    "text": "",
                }
            )

        speaker_count = max(1, min(int(speakers), len(embeddings)))
        if speaker_count == 1:
            labels = [0 for _ in embeddings]
        else:
            labels = AgglomerativeClustering(
                n_clusters=speaker_count,
                metric="cosine",
                linkage="average",
            ).fit_predict(embeddings)

        for turn, label in zip(turns, labels):
            turn["speaker"] = f"SPEAKER_{int(label) + 1}"
        return turns

    def release_gpu(self) -> None:
        if not self.device.startswith("cuda"):
            return
        self.torch.cuda.empty_cache()


class MockSeparator:
    name = "mock_separator"

    def separate(self, audio_path: Path, output_dir: Path, speakers: int) -> list[Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        outputs = []
        for speaker_idx in range(1, max(speakers, 1) + 1):
            target = output_dir / f"{audio_path.stem}_speaker{speaker_idx}.wav"
            if audio_path.exists():
                shutil.copyfile(audio_path, target)
            else:
                target.touch()
            outputs.append(target)
        return outputs


class SpeechBrainSeparator:
    name = "sepformer"

    def __init__(self, model_id: str = "speechbrain/sepformer-whamr16k") -> None:
        cache_dir = _prepare_huggingface_download_env()
        try:
            from speechbrain.inference.separation import SepformerSeparation
        except ImportError:
            from speechbrain.pretrained import SepformerSeparation

        import torch

        self.torch = torch
        _configure_torch_tf32(torch)
        self.model = SepformerSeparation.from_hparams(
            source=model_id,
            savedir=str(cache_dir / "speechbrain" / model_id.split("/")[-1]),
            run_opts={"device": "cuda:0" if torch.cuda.is_available() else "cpu"},
        )

    def separate(self, audio_path: Path, output_dir: Path, speakers: int) -> list[Path]:
        import soundfile as sf

        output_dir.mkdir(parents=True, exist_ok=True)
        sources = self.model.separate_file(path=str(audio_path))
        if hasattr(sources, "detach"):
            sources = sources.detach().cpu()
        if len(sources.shape) == 3:
            sources = sources[0]

        outputs = []
        source_count = min(int(sources.shape[-1]), max(speakers, 1))
        for speaker_idx in range(source_count):
            target = output_dir / f"{audio_path.stem}_speaker{speaker_idx + 1}.wav"
            sf.write(target, sources[:, speaker_idx].numpy(), 16000)
            outputs.append(target)
        return outputs


class ClearVoiceSeparator:
    name = "clearvoice"

    def __init__(self, model_id: str = "MossFormer2_SS_16K") -> None:
        _prepare_huggingface_download_env()
        self._disable_numba_cache_for_clearvoice()

        from clearvoice import ClearVoice

        self.model_id = model_id
        self.model = ClearVoice(
            task="speech_separation",
            model_names=[model_id],
        )

    @staticmethod
    def _disable_numba_cache_for_clearvoice() -> None:
        try:
            import numba
        except ImportError:
            return

        for name in ("jit", "njit", "vectorize", "guvectorize"):
            original = getattr(numba, name, None)
            if original is None or getattr(original, "_overlap_no_cache", False):
                continue

            def no_cache_decorator(*args, _original=original, **kwargs):
                if "cache" in kwargs:
                    kwargs["cache"] = False
                return _original(*args, **kwargs)

            no_cache_decorator._overlap_no_cache = True
            setattr(numba, name, no_cache_decorator)

    def separate(self, audio_path: Path, output_dir: Path, speakers: int) -> list[Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        raw_output_dir = output_dir / "clearvoice_raw"
        self.model(str(audio_path), online_write=True, output_path=str(raw_output_dir))

        outputs = []
        for source_idx, source in enumerate(
            self._find_clearvoice_outputs(raw_output_dir, audio_path.stem),
            start=1,
        ):
            target = output_dir / f"speaker_{source_idx}.wav"
            shutil.copyfile(source, target)
            outputs.append(target)
        shutil.rmtree(raw_output_dir, ignore_errors=True)
        return outputs

    def _find_clearvoice_outputs(
        self,
        raw_output_dir: Path,
        audio_stem: str,
    ) -> list[Path]:
        pattern = re.compile(rf"^{re.escape(audio_stem)}_s(\d+)\.")
        candidates = []
        for path in raw_output_dir.rglob(f"{audio_stem}_s*.*"):
            match = pattern.match(path.name)
            if path.is_file() and match:
                candidates.append((int(match.group(1)), path))
        if candidates:
            return [path for _, path in sorted(candidates)]
        all_outputs = sorted(raw_output_dir.rglob("*"))
        files = [path for path in all_outputs if path.is_file()]
        raise RuntimeError(
            f"ClearVoice did not write separated sources for {audio_stem}. "
            f"Found files: {[path.as_posix() for path in files]}"
        )

    @staticmethod
    def _as_source_array(separated) -> "object":
        import numpy as np

        if isinstance(separated, dict):
            if not separated:
                raise RuntimeError("ClearVoice returned an empty output dictionary.")
            separated = next(iter(separated.values()))
        elif isinstance(separated, (list, tuple)) and len(separated) == 1:
            separated = separated[0]

        sources = np.asarray(separated, dtype=np.float32)
        sources = np.squeeze(sources)
        if sources.ndim == 1:
            raise RuntimeError("ClearVoice returned one waveform instead of separated sources.")
        if sources.ndim == 3:
            sources = sources[:, 0, :]
        if sources.ndim != 2:
            raise RuntimeError(
                f"ClearVoice returned an unsupported source shape: {sources.shape}."
            )
        return sources


class MockLLMRefiner:
    name = "mock_llm_refiner"

    def refine(self, text: str, context: list[str]) -> str:
        prefix = " ".join(context[:2])
        if prefix:
            return f"{text}\n\n[Context used] {prefix}"
        return text


class ApiLLMRefiner:
    """OpenAI-compatible LLM refiner for constrained transcript cleanup."""

    name = "api_llm_refiner"

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self.base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("API LLM refinement requires OPENAI_API_KEY.")

    def refine(self, text: str, context: list[str]) -> str:
        import httpx

        system_prompt = (
            "You are a constrained ASR transcript editor. Improve punctuation, "
            "formatting, and terminology consistency only. Do not add, infer, "
            "recover, or paraphrase words that are not already supported by the "
            "input transcript. The retrieved context is background only, not a "
            "source of missing transcript words. Keep speaker labels and timestamp "
            "order. If a span is unclear, keep the original wording or mark it "
            "uncertain instead of rewriting it. Return JSON with refined_text, "
            "changes, uncertain_spans, and hallucination_risk."
        )
        user_prompt = {
            "retrieved_context": context,
            "transcript": text,
            "output_schema": {
                "refined_text": "string",
                "changes": ["string"],
                "uncertain_spans": ["string"],
                "hallucination_risk": "low|medium|high",
            },
        }
        response = httpx.post(
            f"{self.base_url.rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": json.dumps(user_prompt, ensure_ascii=False),
                    },
                ],
                "temperature": 0,
            },
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        return str(data["choices"][0]["message"]["content"]).strip()


def make_asr(kind: str):
    if kind == "mock":
        return MockASR()
    if kind.startswith("whisper"):
        parts = kind.split(":", 1)
        model_name = parts[1] if len(parts) == 2 else "large-v3"
        return WhisperASR(model_name=model_name)
    if kind.startswith("faster-whisper"):
        parts = kind.split(":", 1)
        model_name = parts[1] if len(parts) == 2 else "large-v3"
        return FasterWhisperASR(model_name=model_name)
    if kind == "funasr":
        return FunASR()
    raise ValueError(f"Unsupported ASR provider: {kind}")


def make_diarizer(kind: str):
    if kind == "mock":
        return MockDiarizer()
    if kind.startswith("pyannote"):
        parts = kind.split(":", 1)
        model_id = parts[1] if len(parts) == 2 else DEFAULT_PYANNOTE_DIARIZATION_MODEL
        return PyannoteDiarizer(model_id=model_id)
    if kind.startswith("speechbrain"):
        parts = kind.split(":", 1)
        model_id = parts[1] if len(parts) == 2 else "speechbrain/spkrec-ecapa-voxceleb"
        return SpeechBrainDiarizer(model_id=model_id)
    raise ValueError(f"Unsupported diarization provider: {kind}")


def make_separator(kind: str):
    if kind == "mock":
        return MockSeparator()
    if kind.startswith("sepformer"):
        parts = kind.split(":", 1)
        model_id = parts[1] if len(parts) == 2 else "speechbrain/sepformer-whamr16k"
        return SpeechBrainSeparator(model_id=model_id)
    if kind.startswith("clearvoice"):
        parts = kind.split(":", 1)
        model_id = parts[1] if len(parts) == 2 else "MossFormer2_SS_16K"
        return ClearVoiceSeparator(model_id=model_id)
    raise ValueError(f"Unsupported separation provider: {kind}")


def make_llm_refiner(kind: str):
    if kind == "mock":
        return MockLLMRefiner()
    if kind == "api":
        return ApiLLMRefiner()
    if kind.startswith("api:"):
        return ApiLLMRefiner(model_name=kind.split(":", 1)[1])
    raise ValueError(f"Unsupported LLM provider: {kind}")
