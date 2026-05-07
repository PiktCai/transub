from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

_nlp_cache: Dict[str, Any] = {}


def _get_nlp(lang: str = "en"):
    """Load spaCy model with lazy initialization."""
    if lang in _nlp_cache:
        return _nlp_cache[lang]

    try:
        import spacy
    except ImportError:
        return None

    model_map = {
        "en": "en_core_web_sm",
        "zh": "zh_core_web_sm",
        "ja": "ja_core_news_sm",
        "de": "de_core_news_sm",
        "fr": "fr_core_news_sm",
        "es": "es_core_news_sm",
        "pt": "pt_core_news_sm",
        "it": "it_core_news_sm",
        "ru": "ru_core_news_sm",
    }

    model_name = model_map.get(lang, "en_core_web_sm")

    try:
        nlp = spacy.load(model_name)
        _nlp_cache[lang] = nlp
        return nlp
    except OSError:
        return None


def _is_valid_clause(doc) -> bool:
    """Check if a span contains a valid clause (subject + verb)."""
    has_subject = any(
        token.dep_ in ("nsubj", "nsubjpass") or token.pos_ == "PRON"
        for token in doc
    )
    has_verb = any(token.pos_ in ("VERB", "AUX") for token in doc)
    return has_subject and has_verb


def _split_by_punctuation(text: str) -> List[str]:
    if not text.strip():
        return []

    parts = re.split(r'(?<=[。！？.!?])\s*', text)
    parts = [p.strip() for p in parts if p.strip()]

    merged = []
    for part in parts:
        if merged and (part.startswith("-") or part.startswith("...")):
            merged[-1] = merged[-1] + " " + part
        elif merged and (merged[-1].endswith("-") or merged[-1].endswith("...")):
            merged[-1] = merged[-1] + " " + part
        else:
            merged.append(part)

    return merged


def _split_by_comma(text: str, nlp) -> List[str]:
    """Split at commas only if the right side forms a valid clause."""
    doc = nlp(text)
    sentences = []

    for sent in doc.sents:
        sent_text = sent.text.strip()
        if not sent_text:
            continue

        comma_positions = [
            i for i, token in enumerate(sent) if token.text == ","
        ]

        if not comma_positions:
            sentences.append(sent_text)
            continue

        best_split = None
        for comma_idx in comma_positions:
            right_start = comma_idx + 1
            right_end = min(right_start + 10, len(sent))
            right_span = sent[right_start:right_end]

            left_end = comma_idx
            left_start = max(0, left_end - 9)
            left_span = sent[left_start:left_end]

            if _is_valid_clause(right_span):
                left_words = [t for t in left_span if t.is_alpha]
                right_words = [t for t in right_span if t.is_alpha]

                if len(left_words) >= 3 and len(right_words) >= 3:
                    best_split = comma_idx

        if best_split is not None:
            left_text = sent[:best_split].text.strip()
            right_text = sent[best_split + 1:].text.strip()
            if left_text:
                sentences.append(left_text)
            if right_text:
                sentences.append(right_text)
        else:
            sentences.append(sent_text)

    return sentences


def _split_by_connectors(text: str, nlp) -> List[str]:
    """Split at conjunctions, validating context with dependency parsing."""
    connectors = {
        "that", "which", "where", "when", "because", "although",
        "while", "if", "unless", "until", "since", "after", "before",
        "and", "but", "or", "yet", "so",
    }

    doc = nlp(text)
    sentences = []

    for sent in doc.sents:
        sent_text = sent.text.strip()
        if not sent_text:
            continue

        split_candidates = []
        for i, token in enumerate(sent):
            if token.text.lower() in connectors:
                if token.pos_ in ("DET", "PRON") and token.head.pos_ == "NOUN":
                    continue

                if token.text in ("'s", "'re", "'ve", "'ll", "'d"):
                    continue

                left_words = [t for t in sent[:i] if t.is_alpha]
                right_words = [t for t in sent[i + 1:] if t.is_alpha]

                if len(left_words) >= 5 and len(right_words) >= 5:
                    split_candidates.append(i)

        if split_candidates:
            split_idx = split_candidates[0]
            left_text = sent[:split_idx].text.strip()
            right_text = sent[split_idx:].text.strip()
            if left_text:
                sentences.append(left_text)
            if right_text:
                sentences.append(right_text)
        else:
            sentences.append(sent_text)

    return sentences


def _split_long_sentences(text: str, nlp, max_tokens: int = 60) -> List[str]:
    """Split long sentences at verb/root boundaries."""
    doc = nlp(text)
    sentences = []

    for sent in doc.sents:
        tokens = [t for t in sent if not t.is_space]

        if len(tokens) <= max_tokens:
            sentences.append(sent.text.strip())
            continue

        split_points = []
        for i, token in enumerate(tokens):
            if token.pos_ in ("VERB", "AUX") or token.dep_ == "ROOT":
                split_points.append(i)

        if not split_points:
            for i in range(0, len(tokens), max_tokens):
                chunk = tokens[i:i + max_tokens]
                chunk_text = " ".join(t.text for t in chunk)
                sentences.append(chunk_text)
            continue

        current_start = 0
        for split_idx in split_points:
            if split_idx - current_start >= 30:
                chunk = tokens[current_start:split_idx]
                chunk_text = " ".join(t.text for t in chunk)
                sentences.append(chunk_text)
                current_start = split_idx

        if current_start < len(tokens):
            chunk = tokens[current_start:]
            chunk_text = " ".join(t.text for t in chunk)
            sentences.append(chunk_text)

    return sentences


def intelligent_split(text: str, lang: str = "en") -> List[str]:
    """Split text intelligently using NLP-based segmentation.

    Pipeline: punctuation -> comma -> connector -> long sentence.
    """
    if not text.strip():
        return []

    nlp = _get_nlp(lang)
    if nlp is None:
        return _split_by_punctuation(text)

    segments = _split_by_punctuation(text)

    comma_split = []
    for segment in segments:
        comma_split.extend(_split_by_comma(segment, nlp))

    connector_split = []
    for segment in comma_split:
        connector_split.extend(_split_by_connectors(segment, nlp))

    final_split = []
    for segment in connector_split:
        final_split.extend(_split_long_sentences(segment, nlp))

    result = []
    for segment in final_split:
        segment = segment.strip()
        if segment and len(segment) > 1:
            result.append(segment)

    return result


def has_spacy_support(lang: str = "en") -> bool:
    """Check if spaCy support is available for the given language."""
    return _get_nlp(lang) is not None


__all__ = ["intelligent_split", "has_spacy_support"]
