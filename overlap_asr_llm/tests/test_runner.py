from pathlib import Path
from dataclasses import replace
import json
import os
import tempfile
import unittest

from overlap_asr_llm.config import LLMRAGSource, load_config
from overlap_asr_llm.cli import _load_env_file
from overlap_asr_llm.io import write_results
from overlap_asr_llm.pipelines import (
    _refined_text_for_scoring,
    _transcribe_speaker_turns,
    run_all,
)
from overlap_asr_llm.providers import (
    DEFAULT_PYANNOTE_DIARIZATION_MODEL,
    ClearVoiceSeparator,
    PyannoteDiarizer,
    Transcript,
    _load_pyannote_pipeline,
    _prepare_huggingface_download_env,
)


class TestRunner(unittest.TestCase):
    def test_mock_runner_produces_four_results_per_sample(self):
        config = load_config(Path("configs/mock.json"))
        config.models.update(
            {"asr": "mock", "diarization": "mock", "separation": "mock", "llm": "mock"}
        )
        config.pipelines[:] = [
            "direct_asr",
            "diarization_asr",
            "separation_asr",
            "llm_rag_refine",
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            config = replace(config, output_dir=Path(tmpdir))
            results = run_all(config)
        self.assertEqual(len(results), len(config.samples) * 4)
        self.assertEqual(
            {result.pipeline for result in results},
            {
                "direct_asr",
                "diarization_asr",
                "separation_asr",
                "llm_rag_refine",
            },
        )

    def test_writer_produces_core_result_files_only(self):
        config = load_config(Path("configs/mock.json"))
        config.models.update(
            {"asr": "mock", "diarization": "mock", "separation": "mock", "llm": "mock"}
        )
        config.pipelines[:] = ["diarization_asr"]
        with tempfile.TemporaryDirectory() as run_tmpdir:
            config = replace(config, output_dir=Path(run_tmpdir))
            results = run_all(config)
        with tempfile.TemporaryDirectory() as tmpdir:
            write_results(results, Path(tmpdir), config.base_dir)
            self.assertTrue((Path(tmpdir) / "results.csv").exists())
            self.assertTrue((Path(tmpdir) / "results.json").exists())
            self.assertTrue((Path(tmpdir) / "run_summary.md").exists())
            self.assertTrue((Path(tmpdir) / "diarization_segments.csv").exists())
            self.assertTrue((Path(tmpdir) / "diarization_segments.json").exists())
            self.assertFalse((Path(tmpdir) / "README.md").exists())
            self.assertFalse((Path(tmpdir) / "qualitative_review_template.csv").exists())
            segments_text = (Path(tmpdir) / "diarization_segments.csv").read_text()
            self.assertIn("SPEAKER1", segments_text)
            self.assertNotIn(str(Path.cwd()), segments_text)
            results_text = (Path(tmpdir) / "results.csv").read_text()
            self.assertIn("text_cer", results_text)
            self.assertIn("text_wer", results_text)
            self.assertIn("speaker_block_cer", results_text)
            self.assertIn("score_basis", results_text)
            summary_text = (Path(tmpdir) / "run_summary.md").read_text()
            self.assertIn("Segments With Text", summary_text)
            self.assertIn("Primary CER", summary_text)
            self.assertIn("Speaker CER", summary_text)
            self.assertNotIn("| CER | WER |", summary_text)

    def test_runner_can_select_direct_asr_only(self):
        config = load_config(Path("configs/mock.json"))
        config.models.update(
            {"asr": "mock", "diarization": "mock", "separation": "mock", "llm": "mock"}
        )
        config.pipelines[:] = ["direct_asr"]
        with tempfile.TemporaryDirectory() as tmpdir:
            config = replace(config, output_dir=Path(tmpdir))
            results = run_all(config)
        self.assertEqual(len(results), len(config.samples))
        self.assertEqual({result.pipeline for result in results}, {"direct_asr"})

    def test_single_config_can_compare_multiple_asr_models(self):
        config = load_config(Path("configs/mock.json"))
        config.models.update(
            {"asr": "mock", "diarization": "mock", "separation": "mock", "llm": "mock"}
        )
        config.pipelines[:] = ["direct_asr"]
        config.asr_models[:] = ["mock", "mock"]
        with tempfile.TemporaryDirectory() as tmpdir:
            config = replace(config, output_dir=Path(tmpdir))
            results = run_all(config)
        self.assertEqual(len(results), len(config.samples) * 2)
        self.assertEqual({result.pipeline for result in results}, {"direct_asr"})
        self.assertEqual([result.model for result in results].count("mock"), len(results))

    def test_llm_rag_only_can_compare_hidden_diarization_sources(self):
        config = load_config(Path("configs/mock.json"))
        config.models.update(
            {"asr": "mock", "diarization": "mock", "separation": "mock", "llm": "mock"}
        )
        config.pipelines[:] = ["llm_rag_refine"]
        config = replace(
            config,
            llm_rag_sources=(
                LLMRAGSource(
                    label="mock_diarization_a",
                    pipeline="diarization_asr",
                    models={"diarization": "mock"},
                ),
                LLMRAGSource(
                    label="mock_diarization_b",
                    pipeline="diarization_asr",
                    models={"diarization": "mock"},
                ),
            ),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            config = replace(config, output_dir=Path(tmpdir))
            results = run_all(config)
        self.assertEqual(len(results), len(config.samples) * 2)
        self.assertEqual({result.pipeline for result in results}, {"llm_rag_refine"})
        self.assertTrue(all("<-" in result.model for result in results))
        self.assertTrue(all("SPEAKER" in result.speaker_labels for result in results))

    def test_refined_text_scoring_strips_json_subtitle_markup(self):
        refined = _refined_text_for_scoring(
            json.dumps(
                {
                    "refined_text": (
                        "00:00:00.000 --> 00:00:01.000 [SPEAKER_1] 你好\n"
                        "00:00:01.000 --> 00:00:02.000 [SPEAKER_2] 世界"
                    )
                }
            )
        )
        self.assertEqual(refined, "你好 世界")

    def test_config_can_set_asr_prompt(self):
        config = load_config(Path("configs/all_pipelines.json"))
        self.assertIsNotNone(config.asr_prompt)
        self.assertIn("简体中文", config.asr_prompt or "")

    def test_cli_env_loader_reads_local_dotenv_without_overriding(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / ".env"
            env_path.write_text(
                "OVERLAP_ASR_LLM_ENABLE_TF32=1\n"
                "EXISTING_OVERLAP_TEST_VALUE=from_file\n",
                encoding="utf-8",
            )
            os.environ["EXISTING_OVERLAP_TEST_VALUE"] = "from_env"
            try:
                os.environ.pop("OVERLAP_ASR_LLM_ENABLE_TF32", None)
                _load_env_file(env_path)
                self.assertEqual(os.environ["OVERLAP_ASR_LLM_ENABLE_TF32"], "1")
                self.assertEqual(os.environ["EXISTING_OVERLAP_TEST_VALUE"], "from_env")
            finally:
                os.environ.pop("OVERLAP_ASR_LLM_ENABLE_TF32", None)
                os.environ.pop("EXISTING_OVERLAP_TEST_VALUE", None)

    def test_sample2_config_extends_shared_base(self):
        config = load_config(Path("configs/direct_asr.json"))
        self.assertEqual(config.pipelines, ["direct_asr"])
        self.assertEqual(config.output_dir.name, "direct_asr")
        self.assertEqual(len(config.samples), 5)
        self.assertTrue(all(sample.reference for sample in config.samples))
        self.assertTrue(all(sample.reference_speakers for sample in config.samples))

    def test_sample2_diarization_model_matches_pyannote_4(self):
        config = load_config(Path("configs/all_pipelines.json"))
        self.assertEqual(
            config.models["diarization"],
            f"pyannote:{DEFAULT_PYANNOTE_DIARIZATION_MODEL}",
        )
        self.assertEqual(
            DEFAULT_PYANNOTE_DIARIZATION_MODEL,
            "pyannote/speaker-diarization-community-1",
        )

    def test_pyannote_diarizer_preloads_waveform_input(self):
        diarizer = PyannoteDiarizer.__new__(PyannoteDiarizer)
        import torch

        waveform = torch.zeros(1, 16000)
        diarizer._load_waveform = lambda audio_path: (waveform, 16000)
        audio_input = diarizer._audio_input(Path("data/samples2/no_overlap.wav"))

        self.assertIs(audio_input["waveform"], waveform)
        self.assertEqual(audio_input["sample_rate"], 16000)
        self.assertEqual(audio_input["uri"], "no_overlap")
        self.assertNotIn("audio", audio_input)

    def test_mock_diarization_adds_speaker_labels(self):
        config = load_config(Path("configs/mock.json"))
        config.models.update(
            {"asr": "mock", "diarization": "mock", "separation": "mock", "llm": "mock"}
        )
        config.pipelines[:] = ["diarization_asr"]
        with tempfile.TemporaryDirectory() as tmpdir:
            config = replace(config, output_dir=Path(tmpdir))
            results = run_all(config)
        self.assertEqual(len(results), len(config.samples))
        self.assertEqual({result.pipeline for result in results}, {"diarization_asr"})
        self.assertTrue(all("SPEAKER" in result.speaker_labels for result in results))

    def test_turn_transcription_skips_short_and_failed_turns(self):
        class FlakyASR:
            name = "flaky_asr"

            def __init__(self):
                self.calls = 0

            def transcribe(self, audio_path, language, prompt=None):
                del audio_path, language, prompt
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("empty model input")
                return Transcript(
                    text="ok",
                    segments=[{"start": 0.0, "end": 0.5, "text": "ok"}],
                )

        turns = [
            {"start": 0.0, "end": 0.05, "speaker": "SPEAKER0"},
            {"start": 1.0, "end": 1.5, "speaker": "SPEAKER1"},
            {"start": 2.0, "end": 2.5, "speaker": "SPEAKER2"},
        ]
        segments = _transcribe_speaker_turns(
            FlakyASR(),
            Path("data/samples2/no_overlap.wav"),
            turns,
            "zh",
        )

        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0]["speaker"], "SPEAKER2")
        self.assertEqual(segments[0]["text"], "ok")

    def test_mock_turn_level_diarization_asr_is_separate_pipeline(self):
        config = load_config(Path("configs/mock.json"))
        config.models.update(
            {"asr": "mock", "diarization": "mock", "separation": "mock", "llm": "mock"}
        )
        config.pipelines[:] = ["diarization_turn_asr"]
        with tempfile.TemporaryDirectory() as tmpdir:
            config = replace(config, output_dir=Path(tmpdir))
            results = run_all(config)
        self.assertEqual(len(results), len(config.samples))
        self.assertEqual({result.pipeline for result in results}, {"diarization_turn_asr"})
        self.assertTrue(all("turn_asr" in result.model for result in results))
        self.assertTrue(all("SPEAKER" in result.speaker_labels for result in results))

    def test_clearvoice_separator_uses_actual_output_source_count(self):
        class FakeClearVoiceModel:
            def __call__(self, audio_path, online_write, output_path):
                raw_dir = Path(output_path) / "MossFormer2_SS_16K"
                raw_dir.mkdir(parents=True)
                stem = Path(audio_path).stem
                (raw_dir / f"{stem}_s2.wav").write_bytes(b"source-two")
                (raw_dir / f"{stem}_s1.wav").write_bytes(b"source-one")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            audio_path = tmp_path / "meeting.wav"
            audio_path.write_bytes(b"audio")
            separator = ClearVoiceSeparator.__new__(ClearVoiceSeparator)
            separator.model = FakeClearVoiceModel()
            outputs = separator.separate(audio_path, tmp_path / "separated", speakers=4)

            self.assertEqual([path.name for path in outputs], ["speaker_1.wav", "speaker_2.wav"])
            self.assertEqual(outputs[0].read_bytes(), b"source-one")
            self.assertEqual(outputs[1].read_bytes(), b"source-two")
            self.assertFalse((tmp_path / "separated" / "clearvoice_raw").exists())

    def test_pyannote_diarizer_accepts_diarize_output_wrapper(self):
        annotation = object()

        class DiarizeOutput:
            speaker_diarization = annotation

        self.assertIs(
            PyannoteDiarizer._speaker_diarization_annotation(DiarizeOutput()),
            annotation,
        )

    def test_pyannote_diarizer_accepts_raw_annotation(self):
        annotation = object()
        self.assertIs(
            PyannoteDiarizer._speaker_diarization_annotation(annotation),
            annotation,
        )

    def test_hf_mirror_endpoint_is_replaced_for_gated_downloads(self):
        original = os.environ.get("HF_ENDPOINT")
        try:
            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
            _prepare_huggingface_download_env()
            self.assertEqual(os.environ["HF_ENDPOINT"], "https://huggingface.co")
        finally:
            if original is None:
                os.environ.pop("HF_ENDPOINT", None)
            else:
                os.environ["HF_ENDPOINT"] = original

    def test_pyannote_loader_uses_pyannote_4_token_api(self):
        class Pipeline:
            @staticmethod
            def from_pretrained(model_id, token=None, cache_dir=None):
                return {"model_id": model_id, "token": token, "cache_dir": cache_dir}

        loaded = _load_pyannote_pipeline(Pipeline, "model", "hf_token", Path("cache"))
        self.assertEqual(loaded["token"], "hf_token")
        self.assertEqual(loaded["cache_dir"], Path("cache"))


if __name__ == "__main__":
    unittest.main()
