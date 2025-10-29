from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict, Iterable, List
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
_CLOSING_WRAPPERS = '"' + "'" + "'" + "\u201c\u201d\uff09)\u3011\u300b\u3009\uff3d]\u300d\u300f\uff1e\uff5d\u3011"
_CJK_LATIN_LEFT_PATTERN = re.compile(r"([\u4e00-\u9fff])([A-Za-z0-9])")
_CJK_LATIN_RIGHT_PATTERN = re.compile(r"([A-Za-z0-9])([\u4e00-\u9fff])")

# Sentence-ending punctuation (highest priority for splits)
_SENTENCE_ENDERS = {'.', '!', '?', '。', '！', '？', '…'}

# Phrase-ending punctuation (medium priority for splits)
_PHRASE_ENDERS = {',', ';', ':', '，', '；', '：', '、'}


def _estimate_display_width(text: str) -> float:
    """Estimate the display width of text for subtitle rendering.
    
    Industry standard: ~42 units for English, but CJK characters are wider.
    This function estimates visual width to ensure single-line display.
    
    Args:
        text: The subtitle text to measure
    
    Returns:
        Estimated display width (1.0 = one standard Latin character)
    """
    width = 0.0
    for char in text:
        if '\u4e00' <= char <= '\u9fff':  # CJK Unified Ideographs
            width += 2.0  # CJK characters are roughly twice as wide
        elif '\u3000' <= char <= '\u303f':  # CJK symbols and punctuation
            width += 2.0
        elif '\uff00' <= char <= '\uffef':  # Fullwidth forms
            width += 2.0
        elif char.isupper():
            width += 1.2  # Uppercase letters are slightly wider
        elif char in 'mwMW':
            width += 1.3  # Wide Latin letters
        elif char in 'il1!|':
            width += 0.5  # Narrow characters
        else:
            width += 1.0  # Standard character
    return width


def _find_sentence_boundary(words: List[Dict[str, Any]], start_idx: int, end_idx: int) -> int | None:
    """Find the best sentence boundary within the given range.
    
    Priority:
    1. Sentence-ending punctuation (. ! ? 。！？)
    2. Phrase-ending punctuation (, ; : ，；：)
    
    Args:
        words: List of word dicts
        start_idx: Start of range to search
        end_idx: End of range to search
    
    Returns:
        Index of the best boundary, or None if no good boundary found
    """
    best_sentence = None
    best_phrase = None
    
    for i in range(start_idx, end_idx + 1):
        word = words[i].get("word", "").strip()
        if not word:
            continue
        
        # Check if word ends with sentence-ending punctuation
        if word[-1] in _SENTENCE_ENDERS:
            best_sentence = i  # Keep looking for the last one
        # Check if word ends with phrase-ending punctuation
        elif word[-1] in _PHRASE_ENDERS and best_sentence is None:
            best_phrase = i
    
    # Prefer sentence boundary over phrase boundary
    return best_sentence if best_sentence is not None else best_phrase


def _split_with_word_timing(
    words: List[Dict[str, Any]],
    max_width: float,
    min_width: float,
    pause_threshold: float = 0.3,
    silence_threshold: float = 2.0,
    remove_silence: bool = True,
    prefer_sentence_boundaries: bool = True,
) -> List[tuple[str, float, float]]:
    """Split using word-level timestamps with professional semantic awareness.
    
    Uses display width estimation, semantic boundaries (sentences/phrases), and
    natural pauses to create professional single-line subtitles.
    
    Args:
        words: List of word dicts with 'word', 'start', 'end' keys
        max_width: Maximum display width (42.0 = industry standard)
        min_width: Minimum display width to avoid too-short lines
        pause_threshold: Minimum gap (seconds) to consider as a natural pause
        silence_threshold: Minimum gap (seconds) to consider as silence
        remove_silence: Whether to skip segments with only silence
        prefer_sentence_boundaries: Prefer splitting at sentence/phrase boundaries
    
    Returns:
        List of (text, start_time, end_time) tuples.
    """
    if not words:
        return []
    
    # Filter out empty words
    words = [w for w in words if w.get("word", "").strip()]
    if not words:
        return []
    
    chunks: List[tuple[str, float, float]] = []
    chunk_start_idx = 0
    
    while chunk_start_idx < len(words):
        current_text = ""
        current_width = 0.0
        current_end_idx = chunk_start_idx
        
        # Candidates for split points with priority scores
        split_candidates: List[tuple[int, float, str]] = []  # (idx, priority, reason)
        
        for i in range(chunk_start_idx, len(words)):
            word = words[i].get("word", "").strip()
            
            # Check for silence gap (always respect these)
            if remove_silence and i > chunk_start_idx:
                prev_end = words[i - 1].get("end", 0.0)
                curr_start = words[i].get("start", 0.0)
                gap = curr_start - prev_end
                
                if gap >= silence_threshold:
                    # Hard stop at silence
                    current_end_idx = i - 1
                    break
            
            # Try adding this word
            if current_text:
                test_text = current_text + " " + word
            else:
                test_text = word
            
            test_width = _estimate_display_width(test_text)
            
            if test_width > max_width:
                # Would exceed max width - need to split
                if current_end_idx > chunk_start_idx:
                    # We have accumulated words, evaluate candidates
                    break
                else:
                    # First word itself exceeds max, include it anyway (rare)
                    current_text = test_text
                    current_width = test_width
                    current_end_idx = i
                    break
            
            # This word fits
            current_text = test_text
            current_width = test_width
            current_end_idx = i
            
            # Evaluate this position as a potential split point
            if current_width >= min_width:
                priority = 0.0
                reason = "neutral"
                
                # 1. Sentence boundary (highest priority)
                if word[-1] in _SENTENCE_ENDERS:
                    priority = 100.0
                    reason = "sentence_end"
                
                # 2. Phrase boundary (high priority)
                elif word[-1] in _PHRASE_ENDERS:
                    priority = 50.0
                    reason = "phrase_end"
                
                # 3. Natural pause (medium priority)
                if i < len(words) - 1:
                    word_end = words[i].get("end", 0.0)
                    next_start = words[i + 1].get("start", 0.0)
                    gap = next_start - word_end
                    
                    if gap >= pause_threshold:
                        priority += 30.0  # Boost priority for natural pauses
                        if reason == "neutral":
                            reason = "pause"
                
                # 4. Optimal width (prefer ~75-90% of max width, but don't aggressively push to max)
                width_ratio = current_width / max_width
                if 0.75 <= width_ratio <= 0.90:
                    priority += 20.0  # Bonus for good length
                elif width_ratio > 0.90:
                    priority += 10.0  # Small bonus for being close to max, but don't force it
                
                if priority > 0 or i == len(words) - 1:
                    split_candidates.append((i, priority, reason))
        
        # Select the best split point
        best_split_idx = current_end_idx
        
        if split_candidates and prefer_sentence_boundaries:
            # Sort by priority (highest first)
            split_candidates.sort(key=lambda x: x[1], reverse=True)
            best_split_idx = split_candidates[0][0]
        
        # Build the chunk
        chunk_words = words[chunk_start_idx : best_split_idx + 1]
        if chunk_words:
            chunk_text = " ".join([w.get("word", "").strip() for w in chunk_words]).strip()
            if chunk_text:
                chunk_start_time = chunk_words[0].get("start", 0.0)
                chunk_end_time = chunk_words[-1].get("end", 0.0)
                chunks.append((chunk_text, chunk_start_time, chunk_end_time))
        
        # Move to next chunk
        chunk_start_idx = best_split_idx + 1
    
    # Post-processing: merge very short last chunk if reasonable
    if len(chunks) >= 2:
        last_width = _estimate_display_width(chunks[-1][0])
        if last_width < min_width:
            prev_end = chunks[-2][2]
            curr_start = chunks[-1][1]
            gap = curr_start - prev_end
            
            # Only merge if not separated by silence and combined width is acceptable
            if gap < silence_threshold:
                prev_text, prev_start, _ = chunks[-2]
                last_text, _, last_end = chunks[-1]
                merged_text = f"{prev_text} {last_text}".strip()
                merged_width = _estimate_display_width(merged_text)
                
                if merged_width <= max_width:
                    chunks[-2] = (merged_text, prev_start, last_end)
                    chunks.pop()
    
    return chunks


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
    words: List[Dict[str, Any]] | None = None  # word-level timestamps

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
            words = segment.get("words")  # Preserve word-level timestamps if available
            lines.append(SubtitleLine(index=idx, start=start, end=end, text=text, words=words))
        return cls(lines=lines)

    def to_serializable(self) -> List[dict]:
        result = []
        for line in self.lines:
            data = {
                "index": line.index,
                "start": line.start,
                "end": line.end,
                "text": line.text,
            }
            if line.words:
                data["words"] = line.words
            result.append(data)
        return result

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
                    words=item.get("words"),  # Preserve word-level data
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

    def refine(
        self,
        max_width: float = 42.0,
        min_width: float = 20.0,
        min_duration: float = 1.2,
        max_cps: float = 20.0,
        pause_threshold: float = 0.3,
        silence_threshold: float = 2.0,
        remove_silence: bool = True,
        prefer_sentence_boundaries: bool = True,
    ) -> "SubtitleDocument":
        """Refine subtitle lines with professional quality.
        
        Args:
            max_width: Maximum display width (42.0 = industry standard for English)
            min_width: Minimum display width (20.0 avoids very short lines)
            min_duration: Minimum duration in seconds (1.2s = professional standard to avoid "flash" subtitles)
            max_cps: Maximum characters per second (20.0 for mixed text, 12-15 for CJK-only)
            pause_threshold: Minimum gap for natural pauses (default 0.3s)
            silence_threshold: Minimum gap to consider as silence (default 2.0s)
            remove_silence: Whether to skip silence segments (default True)
            prefer_sentence_boundaries: Prefer splitting at sentence/phrase boundaries (default True)
        
        Returns:
            Refined subtitle document with intelligent splitting
        """
        
        if not self.lines:
            return SubtitleDocument(lines=[])

        split_lines: List[SubtitleLine] = []
        for line in self.lines:
            # Use word-level timing if available
            if line.words:
                word_chunks = _split_with_word_timing(
                    line.words,
                    max_width,
                    min_width,
                    pause_threshold,
                    silence_threshold,
                    remove_silence,
                    prefer_sentence_boundaries,
                )
                for chunk_text, chunk_start, chunk_end in word_chunks:
                    split_lines.append(
                        SubtitleLine(
                            index=0,
                            start=chunk_start,
                            end=chunk_end,
                            text=chunk_text,
                            words=None,  # Words already consumed
                        )
                    )
            else:
                # Fallback to text-based splitting with even timing distribution
                # For English text without word timestamps, display width ≈ char count
                fallback_max_chars = int(max_width)
                fallback_min_chars = int(min_width)
                chunks = _split_text_for_limits(line.text, fallback_max_chars, fallback_min_chars)
                timings = _allocate_timings(line.start, line.end, len(chunks))
                for chunk_text, (chunk_start, chunk_end) in zip(chunks, timings, strict=True):
                    split_lines.append(
                        SubtitleLine(
                            index=0,
                            start=chunk_start,
                            end=chunk_end,
                            text=chunk_text,
                            words=None,
                        )
                    )

        # Merge neighboring lines when chunks are too short or end on soft punctuation.
        # Use display width for merge decisions
        merged: List[SubtitleLine] = []
        for line in split_lines:
            if not merged:
                merged.append(line)
                continue

            previous = merged[-1]
            combined_text = _combine_text(previous.text, line.text)
            combined_width = _estimate_display_width(combined_text)
            line_width = _estimate_display_width(line.text)
            prev_trimmed = previous.text.rstrip()
            prev_ends_soft = prev_trimmed.endswith(tuple(_SOFT_BREAK_CHARS))
            prev_ends_hard = prev_trimmed.endswith(tuple(_HARD_BREAK_CHARS))
            prev_last_token = prev_trimmed.split()[-1].lower() if prev_trimmed.split() else ""
            prev_ends_connector = (
                any(prev_trimmed.endswith(sfx) for sfx in _CHINESE_CONNECTIVE_SUFFIXES)
                or (prev_trimmed and prev_trimmed[-1] in _CHINESE_CONNECTIVE_CHARS)
                or prev_last_token in _ENGLISH_CONNECTIVES
            )
            
            # Calculate durations and CPS
            line_duration = line.end - line.start
            prev_duration = previous.end - previous.start
            gap_between = line.start - previous.end
            combined_duration = line.end - previous.start
            
            # Calculate CPS (Characters Per Second) for the combined line
            # Use character count (not display width) for CPS calculation
            combined_char_count = len(combined_text)
            combined_cps = combined_char_count / combined_duration if combined_duration > 0 else 0
            
            # Professional standard: CPS should not exceed max_cps to prevent information overload
            cps_ok = combined_cps <= max_cps
            
            # Force merge extremely short lines (orphaned words) even if slightly over max_width
            # A single orphaned word looks worse than a slightly longer line
            ORPHAN_THRESHOLD = 12.0  # Display width threshold for orphaned lines
            MAX_WIDTH_TOLERANCE = 1.25  # Allow 25% overage for orphan merging
            is_orphaned = line_width < ORPHAN_THRESHOLD
            within_tolerance = combined_width <= max_width * MAX_WIDTH_TOLERANCE
            
            # Professional standard: merge short-duration segments to avoid "flash" subtitles
            # But don't merge across natural pauses or long silence gaps (those are intentional splits)
            has_natural_pause = gap_between >= pause_threshold
            has_long_gap = gap_between >= silence_threshold
            # Merge if current line is too short, but only if there's no natural pause between them
            is_short_duration = line_duration < min_duration and not has_natural_pause and not has_long_gap
            
            # All merge conditions must respect CPS limit to prevent information overload
            # For duration-based merging, allow up to 40% width overage to prioritize avoiding "flash" subtitles
            DURATION_WIDTH_TOLERANCE = 1.4
            needs_merge_for_orphan = is_orphaned and within_tolerance and cps_ok
            needs_merge_for_length = line_width < min_width and combined_width <= max_width and cps_ok
            needs_merge_for_duration = is_short_duration and combined_width <= max_width * DURATION_WIDTH_TOLERANCE and cps_ok
            needs_merge_for_soft_break = (
                prev_ends_soft
                and not prev_ends_hard
                and combined_width <= max_width
                and cps_ok
            )
            needs_merge_for_connector = prev_ends_connector and combined_width <= max_width and cps_ok

            if needs_merge_for_orphan or needs_merge_for_length or needs_merge_for_duration or needs_merge_for_soft_break or needs_merge_for_connector:
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
            SubtitleLine(index=idx, start=line.start, end=line.end, text=line.text, words=None)
            for idx, line in enumerate(merged, start=1)
        ]
        return SubtitleDocument(lines=reindexed)

    def apply_offset(self, offset: float) -> "SubtitleDocument":
        """Apply a time offset to all subtitle lines.
        
        Args:
            offset: Seconds to shift (positive delays, negative advances)
        
        Returns:
            New document with adjusted timing
        """
        if offset == 0:
            return self
        
        adjusted: List[SubtitleLine] = []
        for line in self.lines:
            adjusted_line = SubtitleLine(
                index=line.index,
                start=max(0.0, line.start + offset),
                end=max(0.0, line.end + offset),
                text=line.text,
                words=line.words,
            )
            adjusted.append(adjusted_line)
        
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

    def simplify_cjk_punctuation(self) -> "SubtitleDocument":
        """Replace certain punctuation with spaces (common in Chinese subtitles).
        
        Replaces (stopping/pausing punctuation):
        - Commas and periods with double spaces (逗号、句号)
        - Semicolons with double spaces (分号)
        
        Preserves (semantic/connecting punctuation):
        - Question marks, exclamation marks (问号、感叹号)
        - Quotation marks, book title marks (引号、书名号)
        - Parentheses and brackets (括号)
        - Ellipses and dashes (省略号、破折号)
        - Middle dot (间隔号 ·) - used for foreign names like "史蒂夫·乔布斯"
        """
        simplified: List[SubtitleLine] = []
        for line in self.lines:
            text = line.text
            
            # Replace commas with double space for clearer visual separation
            text = re.sub(r"[ \t]*[，,][ \t]*", "  ", text)
            
            # Replace periods with space (but be careful with ellipses and decimals)
            # First protect ellipses and URLs/decimals
            text = text.replace("......", "⟨ELLIPSIS6⟩")
            text = text.replace("......", "⟨ELLIPSIS6⟩")
            text = text.replace("…", "⟨ELLIPSIS⟩")
            text = text.replace("...", "⟨ELLIPSIS3⟩")
            # Protect decimals (e.g., "3.14")
            text = re.sub(r'(\d)\.(\d)', r'\1⟨DOT⟩\2', text)
            # Now replace remaining periods with double space
            text = re.sub(r"[ \t]*[。\.][ \t]*", "  ", text)
            # Restore protected sequences
            text = text.replace("⟨ELLIPSIS6⟩", "……")
            text = text.replace("⟨ELLIPSIS⟩", "…")
            text = text.replace("⟨ELLIPSIS3⟩", "...")
            text = text.replace("⟨DOT⟩", ".")
            
            # Replace semicolons with double space; keep colons for readability
            text = re.sub(r"[ \t]*[；;][ \t]*", "  ", text)
            
            # Normalize multiple exclamation/question marks
            text = re.sub(r'[！!]{2,}', '！', text)
            text = re.sub(r'[？?]{2,}', '？', text)
            text = re.sub(r'[？?！!]+', lambda m: '？！' if '?' in m.group() and '!' in m.group() else m.group()[0], text)
            
            # Clean up excessive spaces while keeping deliberate double spaces
            text = re.sub(r" {3,}", "  ", text).strip()
            
            simplified.append(
                SubtitleLine(
                    index=line.index,
                    start=line.start,
                    end=line.end,
                    text=text,
                )
            )
        return SubtitleDocument(lines=simplified)


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
