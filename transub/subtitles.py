from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Iterable, List
import re

_SOFT_PUNCT_PATTERN = re.compile(r"[，,；;、]")
_HARD_PUNCT_PATTERN = re.compile(r"[。\.!?！？]")
_SOFT_BREAK_CHARS = {"，", ",", "；", ";", "、"}
_HARD_BREAK_CHARS = {"。", ".", "?", "？", "!", "！"}
_SHORT_DANGLERS = {"a", "an", "the", "and", "or", "but", "so", "to", "for", "of", "in", "on", "at", "I", "i"}
_CHINESE_CONNECTIVE_SUFFIXES = ("但是", "不过", "然而", "可是", "所以", "而且")
_CHINESE_CONNECTIVE_CHARS = {"和", "或", "但", "而", "且", "并", "却", "又"}
_ENGLISH_CONNECTIVES = {"and", "or", "but", "so", "yet", "though"}
_TRAILING_PUNCTUATION = "。！？?!?,.;；:：…，、"
_CLOSING_WRAPPERS = "\"'”’）)】》〉］]」』＞｝】"
_CJK_LATIN_LEFT_PATTERN = re.compile(r"([\u4e00-\u9fff])([A-Za-z0-9])")
_CJK_LATIN_RIGHT_PATTERN = re.compile(r"([A-Za-z0-9])([\u4e00-\u9fff])")


def _split_text_for_limits(text: str, max_chars: int, min_chars: int) -> List[str]:
    """Split text into chunks that respect max length while avoiding tiny tails."""

    remaining = text.strip()
    parts: List[str] = []
    while remaining:
        if len(remaining) <= max_chars:
            parts.append(remaining.strip())
            break
        window = remaining[: max_chars]
        split = None
        hard_candidates = [match.end() for match in _HARD_PUNCT_PATTERN.finditer(window)]
        if hard_candidates:
            candidate = hard_candidates[-1]
            if candidate >= min_chars:
                split = candidate
        if split is None:
            soft_candidates = [match.end() for match in _SOFT_PUNCT_PATTERN.finditer(window)]
            if soft_candidates:
                candidate = soft_candidates[-1]
                if candidate >= min_chars:
                    split = candidate
        if split is None:
            space_idx = window.rfind(" ")
            if space_idx >= max(min_chars, 8):
                split = space_idx + 1

        if split is None or split <= 0:
            split = max_chars

        head = remaining[:split].rstrip()
        tail = remaining[split:].lstrip()

        if not head:
            head = remaining[:max_chars].rstrip()
            tail = remaining[max_chars:].lstrip()

        # Avoid leaving a dangling short word at the start of the tail.
        if tail:
            first_word = tail.split(" ", 1)[0]
            if first_word.lower() in _SHORT_DANGLERS and len(head) > min_chars:
                head = f"{head} {first_word}".rstrip()
                tail = tail[len(first_word) :].lstrip()

        if tail:
            original_head = head

            # Move trailing Chinese connector suffixes to the next chunk.
            for suffix in _CHINESE_CONNECTIVE_SUFFIXES:
                if head.endswith(suffix) and len(head) > len(suffix):
                    head = head[: -len(suffix)].rstrip()
                    tail = f"{suffix}{tail}".lstrip()
                    break

            # Single-character Chinese connectors.
            if head and head[-1] in _CHINESE_CONNECTIVE_CHARS:
                connector = head[-1]
                head = head[:-1].rstrip()
                tail = f"{connector}{tail}".lstrip()

            # English connector words.
            if head:
                tokens = head.split()
                if tokens and tokens[-1].lower() in _ENGLISH_CONNECTIVES:
                    connector = tokens.pop()
                    head = " ".join(tokens).rstrip()
                    tail = f"{connector} {tail}".strip()

            if not head:
                head = original_head

        parts.append(head)
        remaining = tail

    # Merge trailing short fragment with the previous chunk.
    if len(parts) >= 2 and len(parts[-1]) < min_chars:
        parts[-2] = (parts[-2].rstrip() + " " + parts[-1]).strip()
        parts.pop()

    return [part.strip() for part in parts if part.strip()]


def _allocate_timings(start: float, end: float, count: int) -> List[tuple[float, float]]:
    """Evenly split timing between child segments (adjust later by adjust_timing)."""

    if count <= 0:
        return []
    if count == 1 or end <= start:
        return [(start, end)] * count
    duration = end - start
    slice_seconds = duration / count
    segments: List[tuple[float, float]] = []
    current = start
    for idx in range(count):
        if idx == count - 1:
            segment_end = end
        else:
            segment_end = current + slice_seconds
        segments.append((current, max(segment_end, current)))
        current = segment_end
    return segments


def _combine_text(left: str, right: str) -> str:
    left = left.rstrip()
    right = right.lstrip()
    if not left:
        return right
    if not right:
        return left

    left_last = left[-1]
    right_first = right[0]

    if left_last.isascii() and right_first.isascii():
        if left_last.isalnum() and right_first.isalnum():
            return f"{left} {right}"
        if left_last in ".?!," and right_first.isalnum():
            return f"{left} {right}"
    return f"{left}{right}"


@dataclass
class SubtitleLine:
    """Represents a single subtitle line with timing."""

    index: int
    start: float  # seconds
    end: float  # seconds
    text: str

    def to_srt_block(self) -> str:
        start_ts = format_timestamp(self.start)
        end_ts = format_timestamp(self.end)
        return f"{self.index}\n{start_ts} --> {end_ts}\n{self.text}\n"

    def to_vtt_block(self) -> str:
        start_ts = format_timestamp(self.start, separator=".")
        end_ts = format_timestamp(self.end, separator=".")
        return f"{start_ts} --> {end_ts}\n{self.text}\n"


@dataclass
class SubtitleDocument:
    """Collection of subtitle lines."""

    lines: List[SubtitleLine]

    def to_srt(self) -> str:
        blocks = [line.to_srt_block().strip("\n") for line in self.lines]
        if not blocks:
            return ""
        return "\n\n".join(blocks) + "\n"

    def to_vtt(self) -> str:
        header = "WEBVTT\n\n"
        blocks = [line.to_vtt_block().strip("\n") for line in self.lines]
        body = "\n\n".join(blocks) if blocks else ""
        return header + body + "\n"

    def chunk(self, size: int) -> Iterable[List[SubtitleLine]]:
        chunk: List[SubtitleLine] = []
        for line in self.lines:
            chunk.append(line)
            if len(chunk) >= size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk

    @classmethod
    def from_whisper_segments(cls, segments: Iterable[dict]) -> "SubtitleDocument":
        lines = []
        for idx, segment in enumerate(segments, start=1):
            text = (segment.get("text") or "").strip()
            start = float(segment.get("start"))
            end = float(segment.get("end"))
            lines.append(SubtitleLine(index=idx, start=start, end=end, text=text))
        return cls(lines=lines)

    def to_serializable(self) -> List[dict]:
        return [
            {
                "index": line.index,
                "start": line.start,
                "end": line.end,
                "text": line.text,
            }
            for line in self.lines
        ]

    @classmethod
    def from_serialized(cls, data: Iterable[dict]) -> "SubtitleDocument":
        lines = []
        for idx, item in enumerate(data, start=1):
            lines.append(
                SubtitleLine(
                    index=int(item.get("index", idx)),
                    start=float(item.get("start", 0.0)),
                    end=float(item.get("end", 0.0)),
                    text=str(item.get("text", "")).strip(),
                )
            )
        return cls(lines=lines)

    @classmethod
    def from_srt(cls, content: str) -> "SubtitleDocument":
        blocks = re.split(r"\n\s*\n", content.strip())
        lines: List[SubtitleLine] = []
        for block in blocks:
            parts = [part.strip("\ufeff") for part in block.strip().splitlines()]
            if len(parts) < 3:
                continue
            try:
                index = int(parts[0].strip())
            except ValueError:
                continue
            timing = parts[1]
            if "-->" not in timing:
                continue
            start_raw, end_raw = [item.strip() for item in timing.split("-->", 1)]
            try:
                start = parse_timestamp(start_raw)
                end = parse_timestamp(end_raw)
            except ValueError:
                continue
            text = "\n".join(parts[2:]).strip()
            lines.append(SubtitleLine(index=index, start=start, end=end, text=text))
        return cls(lines=lines)

    def refine(self, max_chars: int = 60, min_chars: int = 25) -> "SubtitleDocument":
        if not self.lines:
            return SubtitleDocument(lines=[])

        split_lines: List[SubtitleLine] = []
        for line in self.lines:
            chunks = _split_text_for_limits(line.text, max_chars, min_chars)
            timings = _allocate_timings(line.start, line.end, len(chunks))
            for chunk_text, (chunk_start, chunk_end) in zip(chunks, timings, strict=True):
                split_lines.append(
                    SubtitleLine(
                        index=0,
                        start=chunk_start,
                        end=chunk_end,
                        text=chunk_text,
                    )
                )

        # Merge neighboring lines when chunks are too short or end on soft punctuation.
        merged: List[SubtitleLine] = []
        for line in split_lines:
            if not merged:
                merged.append(line)
                continue

            previous = merged[-1]
            combined_text = _combine_text(previous.text, line.text)
            prev_trimmed = previous.text.rstrip()
            prev_ends_soft = prev_trimmed.endswith(tuple(_SOFT_BREAK_CHARS))
            prev_ends_hard = prev_trimmed.endswith(tuple(_HARD_BREAK_CHARS))
            prev_last_token = prev_trimmed.split()[-1].lower() if prev_trimmed.split() else ""
            prev_ends_connector = (
                any(prev_trimmed.endswith(sfx) for sfx in _CHINESE_CONNECTIVE_SUFFIXES)
                or (prev_trimmed and prev_trimmed[-1] in _CHINESE_CONNECTIVE_CHARS)
                or prev_last_token in _ENGLISH_CONNECTIVES
            )
            needs_merge_for_length = len(line.text) < min_chars and len(combined_text) <= max_chars
            needs_merge_for_soft_break = (
                prev_ends_soft
                and not prev_ends_hard
                and len(combined_text) <= max_chars
            )
            needs_merge_for_connector = prev_ends_connector and len(combined_text) <= max_chars

            if needs_merge_for_length or needs_merge_for_soft_break or needs_merge_for_connector:
                merged[-1] = SubtitleLine(
                    index=0,
                    start=previous.start,
                    end=line.end,
                    text=combined_text,
                )
            else:
                merged.append(line)

        # Reindex sequentially to keep downstream translation maps stable.
        reindexed = [
            SubtitleLine(index=idx, start=line.start, end=line.end, text=line.text)
            for idx, line in enumerate(merged, start=1)
        ]
        return SubtitleDocument(lines=reindexed)

    def adjust_timing(
        self,
        trim: float,
        min_duration: float = 0.6,
        gap: float = 0.04,
        offset: float = 0.0,
    ) -> "SubtitleDocument":
        if trim <= 0 and offset == 0:
            return self
        adjusted: List[SubtitleLine] = []
        lines = self.lines
        prev_end_base: float | None = None

        for idx, line in enumerate(lines):
            start = line.start
            end = line.end
            original_duration = max(line.end - line.start, 0.0)
            if original_duration <= 0:
                original_duration = min_duration

            # Respect previous line gap.
            if prev_end_base is not None and start < prev_end_base + gap:
                start = prev_end_base + gap

            # Only trim leading edge if there is slack beyond the required gap.
            if trim > 0 and prev_end_base is not None:
                slack_before = max(0.0, start - max(prev_end_base + gap, line.start))
                start = min(start, line.end)  # safety
                start = min(line.end, start + min(trim, slack_before))

            # Trim trailing edge, prioritizing removal of silence.
            if trim > 0:
                available_back = max(0.0, end - start - min_duration)
                back_trim = min(trim, available_back)
                end -= back_trim

            # Respect next line gap.
            next_start = lines[idx + 1].start if idx + 1 < len(lines) else None
            if next_start is not None and end > next_start - gap:
                end = next_start - gap

            # Ensure we never exceed original boundaries.
            if start < line.start:
                start = line.start
            if end > line.end:
                end = line.end

            # Enforce minimum duration.
            if end - start < min_duration:
                target_end = start + min_duration
                if next_start is not None and target_end > next_start - gap:
                    target_end = min(next_start - gap, line.end)
                    start = max(line.start, target_end - min_duration)
                end = max(target_end, start + 0.01)

            start = max(start, 0.0)
            if end <= start:
                end = start + min_duration

            start_with_offset = start + offset
            end_with_offset = end + offset
            if start_with_offset < 0:
                shift = -start_with_offset
                start_with_offset += shift
                end_with_offset += shift
            if end_with_offset <= start_with_offset:
                end_with_offset = start_with_offset + min_duration

            adjusted_line = SubtitleLine(
                index=line.index,
                start=start_with_offset,
                end=end_with_offset,
                text=line.text,
            )
            adjusted.append(adjusted_line)
            prev_end_base = end

        return SubtitleDocument(lines=adjusted)

    def remove_trailing_punctuation(
        self,
        punctuation: str = _TRAILING_PUNCTUATION,
        closing_wrappers: str = _CLOSING_WRAPPERS,
    ) -> "SubtitleDocument":
        cleaned: List[SubtitleLine] = []
        wrapper_set = set(closing_wrappers)
        for line in self.lines:
            text = line.text.rstrip()
            original = text
            if not text:
                cleaned.append(line)
                continue

            trailing_wrappers: List[str] = []
            while text and text[-1] in wrapper_set:
                trailing_wrappers.append(text[-1])
                text = text[:-1].rstrip()

            stripped = text.rstrip(punctuation).rstrip()
            if not stripped:
                stripped = original
                trailing_wrappers.clear()

            rebuilt = stripped + "".join(reversed(trailing_wrappers))
            cleaned.append(
                SubtitleLine(
                    index=line.index,
                    start=line.start,
                    end=line.end,
                    text=rebuilt,
                )
            )
        return SubtitleDocument(lines=cleaned)

    def normalize_cjk_spacing(self) -> "SubtitleDocument":
        normalized: List[SubtitleLine] = []
        for line in self.lines:
            text = line.text
            text = _CJK_LATIN_LEFT_PATTERN.sub(r"\1 \2", text)
            text = _CJK_LATIN_RIGHT_PATTERN.sub(r"\1 \2", text)
            text = re.sub(r" {2,}", " ", text).strip()
            normalized.append(
                SubtitleLine(
                    index=line.index,
                    start=line.start,
                    end=line.end,
                    text=text,
                )
            )
        return SubtitleDocument(lines=normalized)


def format_timestamp(seconds: float, separator: str = ",") -> str:
    """Format seconds into SRT/VTT timestamp format."""

    if seconds < 0:
        seconds = 0
    delta = timedelta(seconds=seconds)
    total_seconds = int(delta.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    milliseconds = int(delta.microseconds / 1000)
    return f"{hours:02}:{minutes:02}:{secs:02}{separator}{milliseconds:03}"


def parse_timestamp(value: str) -> float:
    """Parse SRT/VTT timestamp into seconds."""

    value = value.strip()
    if not value:
        raise ValueError("Empty timestamp")
    separator = "," if "," in value else "."
    try:
        hours_str, minutes_str, rest = value.split(":", 2)
        seconds_str, millis_str = rest.split(separator)
        hours = int(hours_str)
        minutes = int(minutes_str)
        seconds = int(seconds_str)
        milliseconds = int(millis_str)
    except ValueError as exc:
        raise ValueError(f"Invalid timestamp format: {value}") from exc
    total = hours * 3600 + minutes * 60 + seconds + milliseconds / 1000.0
    return total


__all__ = [
    "SubtitleLine",
    "SubtitleDocument",
    "format_timestamp",
    "parse_timestamp",
]
