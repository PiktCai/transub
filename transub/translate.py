from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from string import Template
from typing import Callable, Dict, Iterable, List, Optional

import requests

from .config import LLMConfig, PipelineConfig
from .subtitles import SubtitleDocument, SubtitleLine

DEFAULT_OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"


class LLMTranslationError(RuntimeError):
    """Raised when the LLM translation step fails."""


@dataclass
class TranslationChunk:
    index: int
    lines: List[SubtitleLine]


class LLMTranslator:
    def __init__(self, config: LLMConfig, pipeline: PipelineConfig) -> None:
        self.config = config
        self.pipeline = pipeline
        if config.api_base:
            base = config.api_base.rstrip("/")
            if base.endswith("chat/completions"):
                self.endpoint = base
            else:
                self.endpoint = f"{base}/chat/completions"
        else:
            self.endpoint = DEFAULT_OPENAI_CHAT_URL
        api_key = os.getenv(config.api_key_env)
        if not api_key:
            raise LLMTranslationError(
                f"Environment variable {config.api_key_env} is not set for LLM translation"
            )
        self.api_key = api_key
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0

    def translate_document(
        self,
        document: SubtitleDocument,
        existing_translations: Optional[Dict[str, str]] = None,
        progress_callback: Optional[Callable[[Dict[str, str]], None]] = None,
    ) -> SubtitleDocument:
        translations: Dict[str, str] = {
            str(int(key)): value.strip()
            for key, value in (existing_translations or {}).items()
        }
        for chunk_index, chunk in enumerate(self._iter_chunks(document), start=1):
            if all(str(line.index) in translations for line in chunk.lines):
                continue
            payload = self._build_payload(chunk_index, chunk)
            attempt = 0
            while True:
                try:
                    response = self._invoke(payload)
                    chunk_translations = self._parse_response(response, chunk)
                    self._update_usage(response.get("usage") or {})
                    update_payload: Dict[str, str] = {}
                    for line, translation in zip(
                        chunk.lines, chunk_translations, strict=True
                    ):
                        translations[str(line.index)] = translation
                        update_payload[str(line.index)] = translation
                    if progress_callback and update_payload:
                        progress_callback(update_payload)
                    break
                except (requests.RequestException, json.JSONDecodeError, LLMTranslationError) as exc:
                    attempt += 1
                    if attempt > self.config.max_retries:
                        raise LLMTranslationError(
                            f"Failed to translate chunk {chunk_index}: {exc}"
                        ) from exc
                    time.sleep(2**attempt * 0.5)
        final_lines: List[SubtitleLine] = []
        for line in document.lines:
            translated_text = translations.get(str(line.index))
            if translated_text is None:
                raise LLMTranslationError(
                    f"Missing translation for line index {line.index}."
                )
            final_lines.append(
                SubtitleLine(
                    index=line.index,
                    start=line.start,
                    end=line.end,
                    text=translated_text,
                )
            )
        self.total_tokens = self.total_prompt_tokens + self.total_completion_tokens
        return SubtitleDocument(lines=final_lines)

    def token_usage(self) -> Dict[str, int]:
        return {
            "prompt": self.total_prompt_tokens,
            "completion": self.total_completion_tokens,
            "total": self.total_tokens,
        }

    def _iter_chunks(self, document: SubtitleDocument) -> Iterable[TranslationChunk]:
        for idx, lines in enumerate(document.chunk(self.config.batch_size), start=1):
            yield TranslationChunk(index=idx, lines=list(lines))

    def _build_payload(self, chunk_index: int, chunk: TranslationChunk) -> dict:
        user_prompt = self._format_prompt(chunk)
        system_prompt = Template(self.pipeline.prompt_preamble).safe_substitute(
            targetLanguage=self.config.target_language,
            style=self.config.style or "",
            model=self.config.model,
            provider=self.config.provider,
        )
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]
        return {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
        }

    def _invoke(self, payload: dict) -> dict:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        response = requests.post(
            self.endpoint,
            headers=headers,
            json=payload,
            timeout=self.config.request_timeout,
        )
        if response.status_code >= 400:
            raise LLMTranslationError(
                f"LLM API error {response.status_code}: {response.text.strip()}"
            )
        try:
            return response.json()
        except ValueError as exc:
            snippet = response.text.strip().splitlines()
            preview = snippet[0][:200] if snippet else ""
            raise LLMTranslationError(
                "LLM API returned non-JSON payload. Partial response: "
                f"{preview}"
            ) from exc

    def _parse_response(self, response: dict, chunk: TranslationChunk) -> List[str]:
        choices = response.get("choices")
        if not choices:
            raise LLMTranslationError("LLM response missing choices")
        message = choices[0].get("message")
        if not message:
            raise LLMTranslationError("LLM response missing message content")
        content = message.get("content", "").strip()
        parsed = self._load_json_response(content)
        if not isinstance(parsed, dict):
            raise LLMTranslationError("Translation response must be a JSON object.")

        expected_keys = [str(line.index) for line in chunk.lines]
        missing_keys = [key for key in expected_keys if key not in parsed]
        extra_keys = [key for key in parsed.keys() if key not in expected_keys]
        if missing_keys:
            raise LLMTranslationError(
                f"Missing translations for keys: {', '.join(missing_keys)}"
            )
        if extra_keys:
            raise LLMTranslationError(
                f"Received unexpected keys in translation: {', '.join(extra_keys)}"
            )

        cleaned: List[str] = []
        for key in expected_keys:
            value = parsed[key]
            if not isinstance(value, str):
                raise LLMTranslationError(
                    f"Translation value for key {key} is not a string"
                )
            cleaned.append(value.strip())
        return cleaned

    def _load_json_response(self, content: str) -> dict:
        """Attempt to parse JSON content, tolerating Markdown fencing."""

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
        if fenced_match:
            fenced_content = fenced_match.group(1)
            try:
                return json.loads(fenced_content)
            except json.JSONDecodeError:
                pass

        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = content[start : end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

        raise LLMTranslationError(
            "LLM response was not valid JSON. Ensure the model returns a pure JSON object."
        )

    def _update_usage(self, usage: Dict[str, object]) -> None:
        prompt = self._coerce_int(
            usage.get("prompt_tokens")
            or usage.get("prompt_token")
            or usage.get("prompt")
        )
        completion = self._coerce_int(
            usage.get("completion_tokens")
            or usage.get("completion_token")
            or usage.get("completion")
        )
        total = self._coerce_int(usage.get("total_tokens") or usage.get("total"))

        if prompt is not None:
            self.total_prompt_tokens += prompt
        if completion is not None:
            self.total_completion_tokens += completion
        if total is not None:
            self.total_tokens = max(self.total_tokens, total)

    @staticmethod
    def _coerce_int(value: object) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _format_prompt(self, chunk: TranslationChunk) -> str:
        batch_payload = {str(line.index): line.text for line in chunk.lines}
        batch_json = json.dumps(
            batch_payload,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        style_instruction = (
            f"Target style: {self.config.style}" if self.config.style else "Maintain a conversational tone."
        )
        return (
            "The following JSON object contains subtitle lines to translate; each key maps to one line.\n"
            f"Target language: {self.config.target_language}.\n"
            f"{style_instruction}\n"
            "Translate only the values, keep the original keys, do not add or remove entries, and do not surround the result with explanations.\n"
            "Return a valid JSON object that can be parsed directly.\n"
            "Content to translate:\n"
            f"{batch_json}\n"
            "Respond with the translated JSON now."
        )


def translate_subtitles(
    document: SubtitleDocument,
    config: LLMConfig,
    pipeline: PipelineConfig,
    existing_translations: Optional[Dict[str, str]] = None,
    progress_callback: Optional[Callable[[Dict[str, str]], None]] = None,
) -> tuple[SubtitleDocument, Dict[str, int]]:
    translator = LLMTranslator(config=config, pipeline=pipeline)
    doc = translator.translate_document(
        document,
        existing_translations=existing_translations,
        progress_callback=progress_callback,
    )
    usage = translator.token_usage()
    return doc, usage


__all__ = ["translate_subtitles", "LLMTranslationError"]
