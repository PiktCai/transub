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

    def test_adjust_timing(self):
        lines = [
            SubtitleLine(index=1, start=1.0, end=2.0, text="Hello"),
            SubtitleLine(index=2, start=3.0, end=4.0, text="World"),
        ]
        doc = SubtitleDocument(lines=lines)

        # Test with a trim value that is smaller than the duration
        adjusted_doc = doc.adjust_timing(trim=0.2)
        self.assertAlmostEqual(adjusted_doc.lines[0].start, 1.0)
        self.assertAlmostEqual(adjusted_doc.lines[0].end, 1.8)
        self.assertAlmostEqual(adjusted_doc.lines[1].start, 3.0)
        self.assertAlmostEqual(adjusted_doc.lines[1].end, 3.8)

        # Test with a trim value that is larger than the duration
        adjusted_doc = doc.adjust_timing(trim=1.2)
        self.assertAlmostEqual(adjusted_doc.lines[0].start, 1.0)
        self.assertAlmostEqual(adjusted_doc.lines[0].end, 1.6)
        self.assertAlmostEqual(adjusted_doc.lines[1].start, 3.0)
        self.assertAlmostEqual(adjusted_doc.lines[1].end, 3.6)

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

        refined_doc = doc.refine(max_chars=30, min_chars=10)
        self.assertEqual(len(refined_doc.lines), 4)
        texts = [line.text for line in refined_doc.lines]
        self.assertEqual(
            texts[:3],
            [
                "This is a very long subtitle",
                "line that needs to be split",
                "into multiple smaller lines.",
            ],
        )
        self.assertTrue(all(len(text) <= 30 for text in texts[:3]))
        self.assertEqual(texts[3], "Short. Merge me.")
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

        refined_doc = doc.refine(max_chars=120, min_chars=15)
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
        refined_doc = doc.refine(max_chars=40, min_chars=10)
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


if __name__ == "__main__":
    unittest.main()
