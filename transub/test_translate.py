import json
import os
import unittest

from transub.config import LLMConfig, PipelineConfig
from transub.subtitles import SubtitleDocument, SubtitleLine
from transub.translate import LLMTranslationError, LLMTranslator


class FakeLLMTranslator(LLMTranslator):
    def __init__(self, responses, *args, **kwargs):
        self._responses = list(responses)
        self.captured_payloads = []
        super().__init__(*args, **kwargs)

    def _invoke(self, payload: dict) -> dict:
        self.captured_payloads.append(payload)
        if not self._responses:
            raise AssertionError("No mock responses left")
        return self._responses.pop(0)


class LLMTranslatorPartialResponseTest(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["TEST_TRANSLATOR_KEY"] = "dummy"
        self.addCleanup(lambda: os.environ.pop("TEST_TRANSLATOR_KEY", None))
        self.pipeline = PipelineConfig()

    @staticmethod
    def _build_document() -> SubtitleDocument:
        lines = [
            SubtitleLine(index=1, start=0.0, end=1.0, text="one"),
            SubtitleLine(index=2, start=1.0, end=2.0, text="two"),
            SubtitleLine(index=3, start=2.0, end=3.0, text="three"),
        ]
        return SubtitleDocument(lines=lines)

    @staticmethod
    def _extract_payload_keys(payload: dict) -> list[str]:
        content = payload["messages"][1]["content"]
        after_marker = content.split("Content to translate:\n", 1)[1]
        json_segment = after_marker.split("\n", 1)[0]
        parsed = json.loads(json_segment)
        return list(parsed.keys())

    def test_translate_document_handles_partial_chunk(self) -> None:
        config = LLMConfig(
            api_key_env="TEST_TRANSLATOR_KEY",
            batch_size=3,
            max_retries=3,
            target_language="es",
        )
        responses = [
            {
                "choices": [
                    {"message": {"content": '{"1": "uno", "2": "dos"}'}}
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
            {
                "choices": [{"message": {"content": '{"3": "tres"}'}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2},
            },
        ]
        translator = FakeLLMTranslator(
            responses=responses,
            config=config,
            pipeline=self.pipeline,
        )
        document = self._build_document()

        translated = translator.translate_document(document)

        self.assertEqual(
            [line.text for line in translated.lines],
            ["uno", "dos", "tres"],
        )
        self.assertEqual(len(translator.captured_payloads), 2)
        self.assertEqual(
            set(self._extract_payload_keys(translator.captured_payloads[0])),
            {"1", "2", "3"},
        )
        self.assertEqual(
            self._extract_payload_keys(translator.captured_payloads[1]),
            ["3"],
        )

    def test_translate_document_respects_retry_limit_for_missing_keys(self) -> None:
        config = LLMConfig(
            api_key_env="TEST_TRANSLATOR_KEY",
            batch_size=3,
            max_retries=1,
            target_language="es",
        )
        responses = [
            {
                "choices": [
                    {"message": {"content": '{"1": "uno", "2": "dos"}'}}
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
            {
                "choices": [
                    {"message": {"content": '{"1": "uno", "2": "dos"}'}}
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
        ]
        translator = FakeLLMTranslator(
            responses=responses,
            config=config,
            pipeline=self.pipeline,
        )
        document = self._build_document()

        with self.assertRaises(LLMTranslationError) as ctx:
            translator.translate_document(document)

        self.assertIn("Missing translations for keys: 3", str(ctx.exception))
        self.assertEqual(len(translator.captured_payloads), 2)
        self.assertEqual(
            set(self._extract_payload_keys(translator.captured_payloads[0])),
            {"1", "2", "3"},
        )
        self.assertEqual(
            self._extract_payload_keys(translator.captured_payloads[1]),
            ["3"],
        )


if __name__ == "__main__":
    unittest.main()
