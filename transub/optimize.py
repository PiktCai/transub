from __future__ import annotations

import json
import os
import re
from typing import Callable, Dict, List, Optional

import requests

from .config import LLMConfig, PipelineConfig
from .subtitles import SubtitleDocument, SubtitleLine
from .smart_retry import SmartRetryHandler
from .cache import get_cached, set_cached, generate_cache_key


ASR_CORRECTION_PROMPT = """# Role: Subtitle Proofreader
You are a professional subtitle proofreader. Your task is to correct ASR (Automatic Speech Recognition) errors in subtitle text.

# Guidelines
1. Fix obvious spelling errors and typos
2. Add missing punctuation (periods, commas, question marks)
3. Fix capitalization at sentence boundaries
4. Do NOT change the meaning or style of the text
5. Do NOT merge or split subtitle lines
6. Preserve the original line indices exactly

# Output Format
Return a JSON object where keys are line indices and values are corrected text.
Example: {"1": "Hello, world!", "2": "How are you?"}"""

POLISH_PROMPT = """# Role: Subtitle Polisher
You are a professional subtitle translator. Your task is to polish translated subtitles for naturalness.

# Guidelines
1. Make the text sound natural and conversational in the target language
2. Fix any awkward phrasing or literal translations
3. Ensure proper punctuation and formatting
4. Do NOT change the meaning
5. Do NOT merge or split subtitle lines
6. Preserve the original line indices exactly

# Output Format
Return a JSON object where keys are line indices and values are polished text.
Example: {"1": "欢迎来到中国", "2": "中国是一个美丽的国家"}"""


class SubtitleOptimizer:
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
            self.endpoint = "https://api.openai.com/v1/chat/completions"
        api_key = os.getenv(config.api_key_env)
        if not api_key:
            raise ValueError(
                f"Environment variable {config.api_key_env} is not set"
            )
        self.api_key = api_key
        self.retry_handler = SmartRetryHandler(enable_circuit_breaker=True)

    def optimize_asr(self, document: SubtitleDocument) -> SubtitleDocument:
        return self._optimize(document, ASR_CORRECTION_PROMPT)

    def polish_translation(self, document: SubtitleDocument) -> SubtitleDocument:
        return self._optimize(document, POLISH_PROMPT)

    def _optimize(self, document: SubtitleDocument, system_prompt: str) -> SubtitleDocument:
        batch_size = self.config.batch_size
        optimized_lines: List[SubtitleLine] = []

        for chunk in document.chunk(batch_size):
            payload = self._build_payload(chunk, system_prompt)
            try:
                response = self.retry_handler.execute_with_retry(self._invoke, payload)
                corrections = self._parse_response(response, chunk)
            except Exception:
                optimized_lines.extend(chunk)
                continue

            for line in chunk:
                corrected = corrections.get(str(line.index), line.text)
                optimized_lines.append(
                    SubtitleLine(
                        index=line.index,
                        start=line.start,
                        end=line.end,
                        text=corrected,
                        words=line.words,
                    )
                )

        return SubtitleDocument(lines=optimized_lines)

    def _build_payload(self, lines: List[SubtitleLine], system_prompt: str) -> dict:
        batch = {str(line.index): line.text for line in lines}
        user_prompt = f"Subtitle lines to optimize:\n{json.dumps(batch, ensure_ascii=False)}"
        return {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
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
            raise RuntimeError(f"LLM API error {response.status_code}: {response.text}")
        result = response.json()
        set_cached(cache_key, result)
        return result

    def _parse_response(self, response: dict, lines: List[SubtitleLine]) -> Dict[str, str]:
        choices = response.get("choices")
        if not choices:
            return {}

        content = choices[0].get("message", {}).get("content", "").strip()
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
            if fenced:
                try:
                    parsed = json.loads(fenced.group(1))
                except json.JSONDecodeError:
                    return {}
            else:
                return {}

        if not isinstance(parsed, dict):
            return {}

        return {str(k): str(v) for k, v in parsed.items()}


def optimize_subtitles(
    document: SubtitleDocument,
    config: LLMConfig,
    pipeline: PipelineConfig,
    mode: str = "asr",
) -> SubtitleDocument:
    optimizer = SubtitleOptimizer(config, pipeline)
    if mode == "asr":
        return optimizer.optimize_asr(document)
    return optimizer.polish_translation(document)


__all__ = ["optimize_subtitles", "SubtitleOptimizer"]
