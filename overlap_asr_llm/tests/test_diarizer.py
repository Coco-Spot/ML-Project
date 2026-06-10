"""Tests for the speaker diarization providers."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
import unittest.mock

from overlap_asr_llm.providers import MockDiarizer, SpeechBrainDiarizer, Transcript


class TestMockDiarizer(unittest.TestCase):
    """The mock diarizer should assign round-robin speaker labels."""

    def setUp(self) -> None:
        self.diarizer = MockDiarizer()

    def test_round_robin_two_speakers(self) -> None:
        transcript = Transcript(
            text="a b c",
            segments=[
                {"start": 0.0, "end": 1.0, "text": "a", "speaker": "UNKNOWN"},
                {"start": 1.0, "end": 2.0, "text": "b", "speaker": "UNKNOWN"},
                {"start": 2.0, "end": 3.0, "text": "c", "speaker": "UNKNOWN"},
            ],
        )
        result = self.diarizer.label(transcript, speakers=2)
        self.assertEqual(
            [seg["speaker"] for seg in result.segments],
            ["SPEAKER1", "SPEAKER2", "SPEAKER1"],
        )

    def test_single_speaker(self) -> None:
        transcript = Transcript(
            text="hello",
            segments=[{"start": 0.0, "end": 2.0, "text": "hello", "speaker": "UNKNOWN"}],
        )
        result = self.diarizer.label(transcript, speakers=1)
        self.assertEqual([seg["speaker"] for seg in result.segments], ["SPEAKER1"])

    def test_accepts_kwargs_for_forward_compat(self) -> None:
        """The mock diarizer should ignore extra keyword arguments so that
        pipelines can pass audio_path without crashing."""
        transcript = Transcript(
            text="x",
            segments=[{"start": 0.0, "end": 1.0, "text": "x", "speaker": "UNKNOWN"}],
        )
        result = self.diarizer.label(
            transcript, speakers=2, audio_path=Path("/nonexistent"), unknown_arg=True
        )
        self.assertIn("SPEAKER", result.segments[0]["speaker"])


class TestSpeechBrainDiarizer(unittest.TestCase):
    """Tests that the SpeechBrain diarizer degrades gracefully when
    dependencies are missing and handles its internal logic correctly."""

    def setUp(self) -> None:
        self.diarizer = SpeechBrainDiarizer()

    def test_fallback_to_mock_when_audio_missing(self) -> None:
        """Without an audio file the diarizer should behave like MockDiarizer."""
        transcript = Transcript(
            text="a b",
            segments=[
                {"start": 0.0, "end": 1.0, "text": "a", "speaker": "UNKNOWN"},
                {"start": 1.0, "end": 2.0, "text": "b", "speaker": "UNKNOWN"},
            ],
        )
        result = self.diarizer.label(transcript, speakers=2)
        self.assertEqual(
            [seg["speaker"] for seg in result.segments],
            ["SPEAKER1", "SPEAKER2"],
        )

    def test_fallback_when_import_fails(self) -> None:
        """If speechbrain is not installed, _lazy_load raises and we fall back
        to mock behaviour."""
        with unittest.mock.patch.object(
            self.diarizer, "_lazy_load", side_effect=ImportError("no speechbrain")
        ):
            transcript = Transcript(
                text="x",
                segments=[{"start": 0.0, "end": 1.0, "text": "x", "speaker": "UNKNOWN"}],
            )
            result = self.diarizer.label(transcript, speakers=2)
            self.assertEqual(result.segments[0]["speaker"], "SPEAKER1")

    def test_fallback_when_lazy_load_returns_none_models(self) -> None:
        """Simulate the _lazy_load succeeding but leaving encoder uninitialised
        (should not happen in practice but verifies the fallback path)."""
        with unittest.mock.patch.object(
            self.diarizer, "_encoder", None
        ):
            transcript = Transcript(
                text="x",
                segments=[{"start": 0.0, "end": 1.0, "text": "x", "speaker": "UNKNOWN"}],
            )
            result = self.diarizer.label(transcript, speakers=2)
            self.assertEqual(result.segments[0]["speaker"], "SPEAKER1")


if __name__ == "__main__":
    unittest.main()
