"""
Free translation backends (Bing/Google) for cost-free translation.
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Optional

import requests

from .subtitles import SubtitleDocument, SubtitleLine


class BingTranslator:
    """Free Bing translation backend."""

    ENDPOINT = "https://api-edge.cognitive.microsofttranslator.com/translate"
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Content-Type": "application/json",
    }

    def translate(self, text: str, source: str, target: str) -> str:
        params = {
            "api-version": "3.0",
            "from": source,
            "to": target,
        }
        body = [{"text": text}]
        try:
            response = requests.post(
                self.ENDPOINT,
                params=params,
                headers=self.HEADERS,
                json=body,
                timeout=30,
            )
            if response.status_code >= 400:
                return text
            result = response.json()
            if result and isinstance(result, list) and result[0].get("translations"):
                return result[0]["translations"][0].get("text", text)
        except Exception:
            pass
        return text


class GoogleTranslator:
    """Free Google translation backend."""

    ENDPOINT = "https://translate.googleapis.com/translate_a/single"

    def translate(self, text: str, source: str, target: str) -> str:
        params = {
            "client": "gtx",
            "sl": source,
            "tl": target,
            "dt": "t",
            "q": text,
        }
        try:
            response = requests.get(
                self.ENDPOINT,
                params=params,
                timeout=30,
            )
            if response.status_code >= 400:
                return text
            result = response.json()
            if result and isinstance(result, list) and result[0]:
                return "".join(item[0] for item in result[0] if item[0])
        except Exception:
            pass
        return text


def translate_document_free(
    document: SubtitleDocument,
    target_language: str,
    source_language: str = "auto",
    backend: str = "bing",
) -> SubtitleDocument:
    """Translate document using free translation backends."""
    if backend == "google":
        translator = GoogleTranslator()
    else:
        translator = BingTranslator()

    translated_lines: List[SubtitleLine] = []
    for line in document.lines:
        translated_text = translator.translate(
            line.text, source_language, target_language
        )
        translated_lines.append(
            SubtitleLine(
                index=line.index,
                start=line.start,
                end=line.end,
                text=translated_text,
                words=line.words,
            )
        )

    return SubtitleDocument(lines=translated_lines)


__all__ = ["translate_document_free", "BingTranslator", "GoogleTranslator"]
