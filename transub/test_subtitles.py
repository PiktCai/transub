from __future__ import annotations

import unittest

from transub.subtitles import SubtitleDocument, SubtitleLine


class SubtitleDocumentTest(unittest.TestCase):
    def test_from_serialized_preserves_index(self):
        serialized_data = [
            {"index": 1, "start": 0.0, "end": 1.0, "text": "Hello"},
            {"index": 2, "start": 1.0, "end": 2.0, "text": "World"},
        ]
        doc = SubtitleDocument.from_serialized(serialized_data)
        self.assertEqual(len(doc.lines), 2)
        self.assertEqual(doc.lines[0].index, 1)
        self.assertEqual(doc.lines[1].index, 2)

    def test_from_serialized_handles_missing_index(self):
        serialized_data = [
            {"start": 0.0, "end": 1.0, "text": "Hello"},
            {"start": 1.0, "end": 2.0, "text": "World"},
        ]
        doc = SubtitleDocument.from_serialized(serialized_data)
        self.assertEqual(len(doc.lines), 2)
        self.assertEqual(doc.lines[0].index, 1)
        self.assertEqual(doc.lines[1].index, 2)

    def test_apply_offset(self):
        lines = [
            SubtitleLine(index=1, start=1.0, end=2.0, text="Hello"),
            SubtitleLine(index=2, start=3.0, end=4.0, text="World"),
        ]
        doc = SubtitleDocument(lines=lines)

        # Test positive offset (delay subtitles)
        delayed_doc = doc.apply_offset(0.5)
        self.assertAlmostEqual(delayed_doc.lines[0].start, 1.5)
        self.assertAlmostEqual(delayed_doc.lines[0].end, 2.5)
        self.assertAlmostEqual(delayed_doc.lines[1].start, 3.5)
        self.assertAlmostEqual(delayed_doc.lines[1].end, 4.5)

        # Test negative offset (advance subtitles)
        advanced_doc = doc.apply_offset(-0.5)
        self.assertAlmostEqual(advanced_doc.lines[0].start, 0.5)
        self.assertAlmostEqual(advanced_doc.lines[0].end, 1.5)
        self.assertAlmostEqual(advanced_doc.lines[1].start, 2.5)
        self.assertAlmostEqual(advanced_doc.lines[1].end, 3.5)
        
        # Test that negative offset doesn't go below 0
        large_negative_doc = doc.apply_offset(-2.0)
        self.assertAlmostEqual(large_negative_doc.lines[0].start, 0.0)
        self.assertAlmostEqual(large_negative_doc.lines[0].end, 0.0)

    def test_remove_trailing_punctuation(self):
        lines = [
            SubtitleLine(index=1, start=1.0, end=2.0, text="Hello."),
            SubtitleLine(index=2, start=3.0, end=4.0, text="World!"),
            SubtitleLine(index=3, start=5.0, end=6.0, text="How are you?"),
            SubtitleLine(index=4, start=7.0, end=8.0, text="I am fine..."),
            SubtitleLine(index=5, start=9.0, end=10.0, text="不错，"),
        ]
        doc = SubtitleDocument(lines=lines)

        adjusted_doc = doc.remove_trailing_punctuation()
        self.assertEqual(adjusted_doc.lines[0].text, "Hello")
        self.assertEqual(adjusted_doc.lines[1].text, "World")
        self.assertEqual(adjusted_doc.lines[2].text, "How are you")
        self.assertEqual(adjusted_doc.lines[3].text, "I am fine")
        self.assertEqual(adjusted_doc.lines[4].text, "不错")

    def test_remove_trailing_punctuation_handles_wrappers(self):
        lines = [
            SubtitleLine(index=1, start=0.0, end=1.0, text="“你好！”"),
            SubtitleLine(index=2, start=1.0, end=2.0, text="(测试。)"),
            SubtitleLine(index=3, start=2.0, end=3.0, text="……"),
        ]
        doc = SubtitleDocument(lines=lines)

        adjusted_doc = doc.remove_trailing_punctuation()
        self.assertEqual(adjusted_doc.lines[0].text, "“你好”")
        self.assertEqual(adjusted_doc.lines[1].text, "(测试)")
        # Entire line is punctuation; keep original fallback.
        self.assertEqual(adjusted_doc.lines[2].text, "……")

    def test_to_srt(self):
        lines = [
            SubtitleLine(index=1, start=1.0, end=2.5, text="Hello"),
            SubtitleLine(index=2, start=3.2, end=4.8, text="World"),
        ]
        doc = SubtitleDocument(lines=lines)
        expected_srt = (
            "1\n00:00:01,000 --> 00:00:02,500\nHello\n\n"
            "2\n00:00:03,200 --> 00:00:04,800\nWorld\n"
        )
        self.assertEqual(doc.to_srt(), expected_srt)

    def test_to_vtt(self):
        lines = [
            SubtitleLine(index=1, start=1.0, end=2.5, text="Hello"),
            SubtitleLine(index=2, start=3.2, end=4.8, text="World"),
        ]
        doc = SubtitleDocument(lines=lines)
        expected_vtt = (
            "WEBVTT\n\n"
            "00:00:01.000 --> 00:00:02.500\nHello\n\n"
            "00:00:03.200 --> 00:00:04.800\nWorld\n"
        )
        self.assertEqual(doc.to_vtt(), expected_vtt)

    def test_from_srt(self):
        srt_content = (
            "1\n00:00:01,000 --> 00:00:02,500\nHello\n\n"
            "2\n00:00:03,200 --> 00:00:04,800\nWorld\n"
        )
        doc = SubtitleDocument.from_srt(srt_content)
        self.assertEqual(len(doc.lines), 2)
        self.assertEqual(doc.lines[0].index, 1)
        self.assertAlmostEqual(doc.lines[0].start, 1.0)
        self.assertAlmostEqual(doc.lines[0].end, 2.5)
        self.assertEqual(doc.lines[0].text, "Hello")
        self.assertEqual(doc.lines[1].index, 2)
        self.assertAlmostEqual(doc.lines[1].start, 3.2)
        self.assertAlmostEqual(doc.lines[1].end, 4.8)
        self.assertEqual(doc.lines[1].text, "World")

    def test_refine(self):
        lines = [
            SubtitleLine(
                index=1,
                start=1.0,
                end=5.0,
                text="This is a very long subtitle line that needs to be split into multiple smaller lines.",
            ),
            SubtitleLine(index=2, start=6.0, end=7.0, text="Short."),
            SubtitleLine(index=3, start=8.0, end=9.0, text="Merge me."),
        ]
        doc = SubtitleDocument(lines=lines)

        refined_doc = doc.refine(max_width=30, min_width=10)
        self.assertEqual(len(refined_doc.lines), 4)
        texts = [line.text for line in refined_doc.lines]
        # With orphan merging, "Short." gets merged with previous line
        self.assertEqual(
            texts[:3],
            [
                "This is a very long subtitle",
                "line that needs to be split",
                "into multiple smaller lines. Short.",
            ],
        )
        # Allow slight overage for orphan merging (up to 25%)
        self.assertTrue(all(len(text) <= 30 * 1.25 for text in texts[:3]))
        self.assertEqual(texts[3], "Merge me.")
        self.assertEqual([line.index for line in refined_doc.lines], [1, 2, 3, 4])

    def test_refine_merges_soft_breaks(self):
        lines = [
            SubtitleLine(
                index=1,
                start=0.0,
                end=2.0,
                text="Compute Module 5 is much smaller than the Raspberry Pi 5,",
            ),
            SubtitleLine(
                index=2,
                start=2.0,
                end=4.0,
                text="so manufacturers can design far more compact carrier boards.",
            ),
        ]
        doc = SubtitleDocument(lines=lines)

        # CPS = 119 chars / 4.0s = 29.75, so we need to allow higher CPS for this test
        refined_doc = doc.refine(max_width=120, min_width=15, max_cps=35.0)
        self.assertEqual(len(refined_doc.lines), 1)
        self.assertTrue(
            refined_doc.lines[0].text.endswith("carrier boards."),
        )

    def test_refine_avoids_trailing_connectives(self):
        lines = [
            SubtitleLine(
                index=1,
                start=0.0,
                end=5.0,
                text="我们可以把主控板和",
            ),
            SubtitleLine(
                index=2,
                start=5.0,
                end=10.0,
                text="接口板一起装进机箱。",
            ),
        ]
        doc = SubtitleDocument(lines=lines)
        refined_doc = doc.refine(max_width=40, min_width=10)
        connectors = ("和", "或", "但", "而", "且", "并", "但是", "不过", "然而", "可是")
        for line in refined_doc.lines[:-1]:
            trimmed = line.text.rstrip()
            self.assertTrue(trimmed)
            self.assertFalse(trimmed.endswith(connectors))

    def test_normalize_cjk_spacing(self):
        lines = [
            SubtitleLine(index=1, start=0.0, end=1.0, text="这台GPU很快"),
            SubtitleLine(index=2, start=1.0, end=2.0, text="支持AI推理2024版"),
        ]
        doc = SubtitleDocument(lines=lines)
        normalized = doc.normalize_cjk_spacing()
        self.assertEqual(normalized.lines[0].text, "这台 GPU 很快")
        self.assertEqual(normalized.lines[1].text, "支持 AI 推理 2024 版")

    def test_simplify_cjk_punctuation_basic(self):
        """Test basic comma and period replacement."""
        lines = [
            SubtitleLine(index=1, start=0.0, end=1.0, text="你好，世界。"),
            SubtitleLine(index=2, start=1.0, end=2.0, text="Hello, world."),
        ]
        doc = SubtitleDocument(lines=lines)
        simplified = doc.simplify_cjk_punctuation()
        self.assertEqual(simplified.lines[0].text, "你好  世界")
        self.assertEqual(simplified.lines[1].text, "Hello  world")

    def test_simplify_cjk_punctuation_preserves_important_marks(self):
        """Test that important punctuation is preserved."""
        lines = [
            SubtitleLine(index=1, start=0.0, end=1.0, text="你好！世界？"),
            SubtitleLine(index=2, start=1.0, end=2.0, text='这是《红楼梦》的"书名号"'),
            SubtitleLine(index=3, start=2.0, end=3.0, text="括号(测试)和【方括号】"),
            SubtitleLine(index=4, start=3.0, end=4.0, text="史蒂夫·乔布斯和J·K·罗琳"),
        ]
        doc = SubtitleDocument(lines=lines)
        simplified = doc.simplify_cjk_punctuation()
        # Exclamation and question marks preserved
        self.assertEqual(simplified.lines[0].text, "你好！世界？")
        # Book title marks and quotes preserved
        self.assertEqual(simplified.lines[1].text, '这是《红楼梦》的"书名号"')
        # Parentheses preserved
        self.assertEqual(simplified.lines[2].text, "括号(测试)和【方括号】")
        # Middle dot preserved (used for foreign names)
        self.assertEqual(simplified.lines[3].text, "史蒂夫·乔布斯和J·K·罗琳")

    def test_simplify_cjk_punctuation_handles_ellipsis(self):
        """Test that ellipses are preserved."""
        lines = [
            SubtitleLine(index=1, start=0.0, end=1.0, text="等等…"),
            SubtitleLine(index=2, start=1.0, end=2.0, text="然后......就结束了。"),
            SubtitleLine(index=3, start=2.0, end=3.0, text="Wait..."),
        ]
        doc = SubtitleDocument(lines=lines)
        simplified = doc.simplify_cjk_punctuation()
        # Chinese ellipsis preserved
        self.assertEqual(simplified.lines[0].text, "等等…")
        # Six-dot ellipsis preserved, but trailing period removed
        self.assertEqual(simplified.lines[1].text, "然后……就结束了")
        # English ellipsis preserved
        self.assertEqual(simplified.lines[2].text, "Wait...")

    def test_simplify_cjk_punctuation_handles_decimals(self):
        """Test that decimal numbers are preserved."""
        lines = [
            SubtitleLine(index=1, start=0.0, end=1.0, text="圆周率是3.14159。"),
            SubtitleLine(index=2, start=1.0, end=2.0, text="价格是99.99元。"),
        ]
        doc = SubtitleDocument(lines=lines)
        simplified = doc.simplify_cjk_punctuation()
        # Decimal points preserved, trailing periods removed
        self.assertEqual(simplified.lines[0].text, "圆周率是3.14159")
        self.assertEqual(simplified.lines[1].text, "价格是99.99元")

    def test_simplify_cjk_punctuation_handles_semicolon_colon(self):
        """Semicolons become double spaces while colons are preserved."""
        lines = [
            SubtitleLine(index=1, start=0.0, end=1.0, text="注意：这很重要；记住了。"),
            SubtitleLine(index=2, start=1.0, end=2.0, text="Note: very important; remember."),
        ]
        doc = SubtitleDocument(lines=lines)
        simplified = doc.simplify_cjk_punctuation()
        self.assertEqual(simplified.lines[0].text, "注意：这很重要  记住了")
        self.assertEqual(simplified.lines[1].text, "Note: very important  remember")

    def test_simplify_cjk_punctuation_normalizes_repeated_marks(self):
        """Test that repeated exclamation/question marks are normalized."""
        lines = [
            SubtitleLine(index=1, start=0.0, end=1.0, text="什么！！！"),
            SubtitleLine(index=2, start=1.0, end=2.0, text="真的吗？？？"),
            SubtitleLine(index=3, start=2.0, end=3.0, text="不会吧？！"),
        ]
        doc = SubtitleDocument(lines=lines)
        simplified = doc.simplify_cjk_punctuation()
        # Multiple exclamation marks normalized to one
        self.assertEqual(simplified.lines[0].text, "什么！")
        # Multiple question marks normalized to one
        self.assertEqual(simplified.lines[1].text, "真的吗？")
        # Mixed marks kept as interrobang
        self.assertEqual(simplified.lines[2].text, "不会吧？")

    def test_simplify_cjk_punctuation_cleans_spaces(self):
        """Test that spaces are capped at a double space."""
        lines = [
            SubtitleLine(index=1, start=0.0, end=1.0, text="你好，  世界。  再见。"),
        ]
        doc = SubtitleDocument(lines=lines)
        simplified = doc.simplify_cjk_punctuation()
        # Capped at double spaces
        self.assertEqual(simplified.lines[0].text, "你好  世界  再见")


    def test_refine_with_word_timestamps(self):
        """Test that refine() uses word-level timestamps when available."""
        words = [
            {"word": "This", "start": 0.0, "end": 0.2},
            {"word": "is", "start": 0.2, "end": 0.4},
            {"word": "a", "start": 0.4, "end": 0.5},
            {"word": "very", "start": 0.5, "end": 0.8},
            {"word": "long", "start": 0.8, "end": 1.0},
            {"word": "sentence", "start": 1.0, "end": 1.5},
            {"word": "that", "start": 1.5, "end": 1.7},
            {"word": "needs", "start": 1.7, "end": 2.0},
            {"word": "splitting", "start": 2.0, "end": 2.5},
        ]
        lines = [
            SubtitleLine(
                index=1, 
                start=0.0, 
                end=2.5, 
                text="This is a very long sentence that needs splitting",
                words=words
            )
        ]
        doc = SubtitleDocument(lines=lines)
        refined = doc.refine(max_width=25, min_width=10)
        
        # Should split using word boundaries
        self.assertGreater(len(refined.lines), 1)
        # First chunk should use actual word timing, not evenly distributed
        self.assertAlmostEqual(refined.lines[0].start, 0.0, places=1)
        # Timing should come from actual word timestamps
        self.assertLess(refined.lines[0].end, 1.5)  # Not the midpoint of 2.5

    def test_refine_without_word_timestamps(self):
        """Test that refine() falls back gracefully without word timestamps."""
        lines = [
            SubtitleLine(
                index=1, 
                start=0.0, 
                end=2.5, 
                text="This is a very long sentence that needs splitting",
                words=None  # No word timestamps
            )
        ]
        doc = SubtitleDocument(lines=lines)
        refined = doc.refine(max_width=25, min_width=10)
        
        # Should still split, but with even timing distribution
        self.assertGreater(len(refined.lines), 1)

    def test_word_timestamps_preserved_in_serialization(self):
        """Test that word timestamps are saved and loaded correctly."""
        words = [
            {"word": "Hello", "start": 0.0, "end": 0.5},
            {"word": "world", "start": 0.6, "end": 1.0},
        ]
        lines = [
            SubtitleLine(index=1, start=0.0, end=1.0, text="Hello world", words=words)
        ]
        doc = SubtitleDocument(lines=lines)
        
        # Serialize
        serialized = doc.to_serializable()
        self.assertIn("words", serialized[0])
        self.assertEqual(len(serialized[0]["words"]), 2)
        
        # Deserialize
        restored = SubtitleDocument.from_serialized(serialized)
        self.assertEqual(len(restored.lines), 1)
        self.assertIsNotNone(restored.lines[0].words)
        self.assertEqual(len(restored.lines[0].words), 2)
        self.assertEqual(restored.lines[0].words[0]["word"], "Hello")

    def test_smart_splitting_with_pauses(self):
        """Test that long sentences split at natural pauses."""
        # Create words with clear pauses
        words = [
            {"word": "Hello", "start": 0.0, "end": 0.3},
            {"word": "world", "start": 0.35, "end": 0.7},
            # Natural pause here (0.7 -> 1.2 = 0.5s gap)
            {"word": "this", "start": 1.2, "end": 1.4},
            {"word": "is", "start": 1.45, "end": 1.6},
            {"word": "a", "start": 1.65, "end": 1.7},
            {"word": "test", "start": 1.75, "end": 2.0},
        ]
        
        line = SubtitleLine(
            index=1, start=0.0, end=2.0, text="Hello world this is a test", words=words
        )
        doc = SubtitleDocument(lines=[line])
        
        # Split with pause detection
        refined = doc.refine(
            max_width=20,  # Force a split
            min_width=5,
            pause_threshold=0.4,  # Will detect the 0.5s gap
            silence_threshold=2.0,
            remove_silence=False,
        )
        
        # Should split at the natural pause
        self.assertGreater(len(refined.lines), 1, "Should split into multiple lines")
        # First chunk should be "Hello world"
        self.assertIn("Hello", refined.lines[0].text)
        self.assertIn("world", refined.lines[0].text)
        # Second chunk should contain "this is a test"
        self.assertIn("this", refined.lines[1].text)

    def test_silence_removal(self):
        """Test that long silence segments are removed."""
        # Create words with a long silence gap
        words = [
            {"word": "Hello", "start": 0.0, "end": 0.5},
            {"word": "world", "start": 0.6, "end": 1.0},
            # Long silence here (1.0 -> 5.0 = 4s gap)
            {"word": "after", "start": 5.0, "end": 5.3},
            {"word": "silence", "start": 5.4, "end": 5.8},
        ]
        
        line = SubtitleLine(
            index=1,
            start=0.0,
            end=5.8,
            text="Hello world after silence",
            words=words,
        )
        doc = SubtitleDocument(lines=[line])
        
        # Refine with silence removal enabled
        refined = doc.refine(
            max_width=60,
            min_width=5,
            pause_threshold=0.3,
            silence_threshold=2.0,  # Will detect the 4s gap
            remove_silence=True,
        )
        
        # Should create separate chunks before and after silence
        self.assertGreaterEqual(len(refined.lines), 2, "Should split around silence")
        
        # Check timing: first chunk should end before silence
        self.assertLess(refined.lines[0].end, 2.0, "First chunk should end before silence")
        # Second chunk should start after silence
        self.assertGreater(refined.lines[1].start, 4.0, "Second chunk should start after silence")

    def test_no_silence_removal_when_disabled(self):
        """Test that silence removal can be disabled."""
        words = [
            {"word": "Hello", "start": 0.0, "end": 0.5},
            # Long silence here (0.5 -> 3.0 = 2.5s gap)
            {"word": "world", "start": 3.0, "end": 3.5},
        ]
        
        line = SubtitleLine(
            index=1, start=0.0, end=3.5, text="Hello world", words=words
        )
        doc = SubtitleDocument(lines=[line])
        
        # Refine with silence removal disabled
        refined = doc.refine(
            max_width=60,
            min_width=5,
            pause_threshold=0.3,
            silence_threshold=2.0,
            remove_silence=False,  # Disabled
        )
        
        # Should not split at silence when disabled
        # (might still be one line if text is short enough)
        if len(refined.lines) > 1:
            # If it did split (for other reasons), the gap should be preserved
            gap = refined.lines[1].start - refined.lines[0].end
            # Gap should be close to original (not removed)
            self.assertGreater(gap, 1.0, "Silence gap should be preserved")


    def test_display_width_estimation(self):
        """Test that display width is correctly estimated for mixed content."""
        from transub.subtitles import _estimate_display_width
        
        # Pure English (accounting for 'H' being uppercase = 1.2)
        english_width = _estimate_display_width("Hello World")
        self.assertGreater(english_width, 9.0)
        self.assertLess(english_width, 12.0)
        
        # Pure CJK (each char = 2.0)
        cjk_width = _estimate_display_width("你好世界")
        self.assertAlmostEqual(cjk_width, 8.0, delta=0.5)
        
        # CJK should be wider than same-length Latin
        self.assertGreater(cjk_width, _estimate_display_width("test"))
        
        # Mixed content
        mixed = "Hello 你好"  # Latin + CJK
        mixed_width = _estimate_display_width(mixed)
        self.assertGreater(mixed_width, 9.0)
        self.assertLess(mixed_width, 15.0)
    
    def test_semantic_splitting_with_sentence_boundaries(self):
        """Test that subtitles prefer splitting at sentence boundaries."""
        words = [
            {"word": "Hello", "start": 0.0, "end": 0.5},
            {"word": "world.", "start": 0.6, "end": 1.0},
            {"word": "How", "start": 1.5, "end": 1.8},
            {"word": "are", "start": 1.9, "end": 2.1},
            {"word": "you", "start": 2.2, "end": 2.4},
            {"word": "today?", "start": 2.5, "end": 2.9},
        ]
        
        line = SubtitleLine(
            index=1,
            start=0.0,
            end=2.9,
            text="Hello world. How are you today?",
            words=words,
        )
        doc = SubtitleDocument(lines=[line])
        
        # Split with low max_width to force splitting
        refined = doc.refine(
            max_width=15.0,  # Force a split
            min_width=5.0,
            prefer_sentence_boundaries=True,
        )
        
        # Should split at sentence boundary (after "world.")
        self.assertGreaterEqual(len(refined.lines), 2)
        # First line should end with sentence boundary
        first_line_ends_with_period = refined.lines[0].text.rstrip().endswith('.')
        self.assertTrue(first_line_ends_with_period, 
                       f"Expected first line to end with '.', got: {refined.lines[0].text}")
    
    def test_cjk_width_aware_splitting(self):
        """Test that CJK content is split based on display width."""
        # Create a long Chinese sentence
        words = [
            {"word": "这是", "start": 0.0, "end": 0.5},
            {"word": "一个", "start": 0.6, "end": 1.0},
            {"word": "测试", "start": 1.1, "end": 1.5},
            {"word": "句子", "start": 1.6, "end": 2.0},
        ]
        
        line = SubtitleLine(
            index=1,
            start=0.0,
            end=2.0,
            text="这是 一个 测试 句子",
            words=words,
        )
        doc = SubtitleDocument(lines=[line])
        
        # Each Chinese char = 2.0 width, so "这是 一个 测试 句子" ≈ 20 width
        # Split with max_width=12 should create multiple lines
        refined = doc.refine(
            max_width=12.0,
            min_width=4.0,
        )
        
        # Should split into at least 2 lines
        self.assertGreaterEqual(len(refined.lines), 2)


if __name__ == "__main__":
    unittest.main()
