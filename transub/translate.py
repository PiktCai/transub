from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Callable, Dict, Iterable, List, Optional

import requests

from .auth import AuthManager, resolve_llm_credentials
from .config import LLMConfig, PipelineConfig
from .smart_retry import SmartRetryHandler, ErrorType
from .subtitles import SubtitleDocument, SubtitleLine
from .cache import get_cached, set_cached, generate_cache_key
from json_repair import repair_json

DEFAULT_OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"


class LLMTranslationError(RuntimeError):
    """Raised when the LLM translation step fails."""


@dataclass
class TranslationChunk:
    index: int
    lines: List[SubtitleLine]


@dataclass
class ParsedChunkResult:
    translations: Dict[str, str]
    missing_keys: List[str]
    unexpected_keys: List[str]


class LLMTranslator:
    def __init__(self, config: LLMConfig, pipeline: PipelineConfig) -> None:
        self.config = config
        self.pipeline = pipeline
        credentials = resolve_llm_credentials(
            provider=config.provider,
            api_key_env=config.api_key_env,
        )
        api_base = config.api_base or credentials.api_base
        if api_base:
            base = api_base.rstrip("/")
            if base.endswith("chat/completions"):
                self.endpoint = base
            else:
                self.endpoint = f"{base}/chat/completions"
        else:
            self.endpoint = DEFAULT_OPENAI_CHAT_URL
        if not credentials.api_key:
            auth_path = AuthManager.default_path()
            raise LLMTranslationError(
                f"LLM credentials are missing. Set {config.api_key_env} or save "
                f"{config.provider!r} credentials in {auth_path}."
            )
        self.api_key = credentials.api_key
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        
        # Initialize smart retry handler
        self.retry_handler = SmartRetryHandler(
            enable_circuit_breaker=True,
        )
        
        # Load glossary
        self.glossary = self._load_glossary()

    def _load_glossary(self) -> Dict[str, str]:
        """Load glossary from file if specified."""
        glossary_path = self.pipeline.glossary_path
        if not glossary_path:
            return {}
        
        path = Path(glossary_path)
        if not path.exists():
            return {}
        
        try:
            if path.suffix.lower() == '.json':
                with path.open('r', encoding='utf-8') as f:
                    return json.load(f)
            elif path.suffix.lower() == '.csv':
                import csv
                glossary = {}
                with path.open('r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    for row in reader:
                        if len(row) >= 2:
                            glossary[row[0].strip()] = row[1].strip()
                return glossary
        except Exception:
            pass
        return {}

    def _get_context(self, document: SubtitleDocument, current_index: int) -> tuple[str, str]:
        """Get previous and next context lines."""
        window = self.config.context_window
        if window == 0:
            return "", ""
        
        lines = document.lines
        current_pos = None
        for i, line in enumerate(lines):
            if line.index == current_index:
                current_pos = i
                break
        
        if current_pos is None:
            return "", ""
        
        prev_lines = []
        for i in range(max(0, current_pos - window), current_pos):
            prev_lines.append(lines[i].text)
        
        next_lines = []
        for i in range(current_pos + 1, min(len(lines), current_pos + window + 1)):
            next_lines.append(lines[i].text)
        
        return " | ".join(prev_lines), " | ".join(next_lines)

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
            pending_lines = [
                line for line in chunk.lines if str(line.index) not in translations
            ]
            if not pending_lines:
                continue
            attempt = 0
            while pending_lines:
                pending_chunk = TranslationChunk(index=chunk.index, lines=pending_lines)
                payload = self._build_payload(chunk_index, pending_chunk, document)
                try:
                    response = self.retry_handler.execute_with_retry(self._invoke, payload)
                    result = self._parse_response(response, pending_chunk)
                    self._update_usage(response.get("usage") or {})
                except LLMTranslationError as exc:
                    attempt += 1
                    if attempt > 3:
                        raise
                    error_content = response.get('choices', [{}])[0].get('message', {}).get('content', '')
                    if error_content:
                        payload['messages'].append({"role": "assistant", "content": error_content})
                        payload['messages'].append({"role": "user", "content": f"Error: {exc}\nPlease fix the JSON and provide the missing translations."})
                    continue
                
                update_payload: Dict[str, str] = {}
                for line in pending_lines:
                    key = str(line.index)
                    translation = result.translations.get(key)
                    if translation is not None:
                        translations[key] = translation
                        update_payload[key] = translation

                if progress_callback and update_payload:
                    progress_callback(update_payload)

                unexpected = [
                    key for key in result.unexpected_keys if key not in translations
                ]
                if unexpected:
                    raise LLMTranslationError(
                        f"Received unexpected keys in translation: {', '.join(unexpected)}"
                    )

                if not result.missing_keys:
                    break

                pending_lines = [
                    line for line in pending_lines if str(line.index) in result.missing_keys
                ]
                attempt += 1
                if attempt > self.config.max_retries:
                    raise LLMTranslationError(
                        f"Failed to translate chunk {chunk_index}: Missing translations for keys: {', '.join(result.missing_keys)}"
                    )
                time.sleep(min(2**attempt * 0.2, 2.0))
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

    def _build_payload(self, chunk_index: int, chunk: TranslationChunk, document: SubtitleDocument) -> dict:
        user_prompt = self._format_prompt(chunk, document)
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
        cache_key = generate_cache_key(payload)
        cached = get_cached(cache_key)
        if cached:
            return cached

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
            result = response.json()
            set_cached(cache_key, result)
            return result
        except ValueError as exc:
            snippet = response.text.strip().splitlines()
            preview = snippet[0][:200] if snippet else ""
            raise LLMTranslationError(
                "LLM API returned non-JSON payload. Partial response: "
                f"{preview}"
            ) from exc

    def _parse_response(self, response: dict, chunk: TranslationChunk) -> ParsedChunkResult:
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
        translations: Dict[str, str] = {}

        for key in expected_keys:
            if key not in parsed:
                continue
            value = parsed[key]
            if not isinstance(value, str):
                raise LLMTranslationError(
                    f"Translation value for key {key} is not a string"
                )
            translations[key] = value.strip()

        unexpected_keys = [key for key in parsed.keys() if key not in expected_keys]
        missing_keys = [key for key in expected_keys if key not in translations]
        
        return ParsedChunkResult(
            translations=translations,
            missing_keys=missing_keys,
            unexpected_keys=unexpected_keys,
        )

    def _load_json_response(self, content: str) -> dict:

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        try:
            repaired = repair_json(content)
            return json.loads(repaired)
        except Exception:
            pass

        fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
        if fenced_match:
            fenced_content = fenced_match.group(1)
            try:
                return json.loads(fenced_content)
            except json.JSONDecodeError:
                try:
                    repaired = repair_json(fenced_content)
                    return json.loads(repaired)
                except Exception:
                    pass

        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = content[start : end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                try:
                    repaired = repair_json(candidate)
                    return json.loads(repaired)
                except Exception:
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

    def _format_prompt(self, chunk: TranslationChunk, document: SubtitleDocument) -> str:
        batch_payload = {str(line.index): line.text for line in chunk.lines}
        batch_json = json.dumps(
            batch_payload,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        style_instruction = (
            f"Target style: {self.config.style}" if self.config.style else "Maintain a conversational tone."
        )
        
        prev_context, next_context = self._get_context(document, chunk.lines[0].index)
        
        context_section = ""
        if prev_context or next_context:
            context_section = "\n### Context\n"
            if prev_context:
                context_section += f"Previous: {prev_context}\n"
            if next_context:
                context_section += f"Next: {next_context}\n"
        
        glossary_section = ""
        if self.glossary:
            glossary_section = "\n### Glossary (Must follow)\n"
            for src, tgt in self.glossary.items():
                glossary_section += f"- {src} -> {tgt}\n"
        
        return (
            "The following JSON object contains subtitle lines to translate; each key maps to one line.\n"
            f"Target language: {self.config.target_language}.\n"
            f"{style_instruction}\n"
            f"{context_section}"
            f"{glossary_section}"
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
