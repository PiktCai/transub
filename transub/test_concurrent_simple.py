"""
Simple test for concurrent translation functionality
"""

from __future__ import annotations

import asyncio
import time
import unittest
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, List

from transub.concurrent_translate import (
    ConcurrentTranslationManager,
    TranslationTask,
    TranslationResult,
    RateLimiter,
)
from transub.subtitles import SubtitleDocument, SubtitleLine
from transub.config import LLMConfig, PipelineConfig
from transub.translate import LLMTranslationError, TranslationChunk


class TestConcurrentTranslationSimple(unittest.TestCase):
    """Simple test for concurrent translation functionality"""
    
    def setUp(self):
        self.manager = ConcurrentTranslationManager(max_concurrency=3)
    
    def test_task_creation_and_results(self):
        """Test basic task creation and result handling"""
        # Create a simple translation task
        lines = [
            SubtitleLine(index=1, start=0.0, end=1.0, text="Hello"),
            SubtitleLine(index=2, start=1.0, end=2.0, text="World"),
        ]
        chunk = TranslationChunk(index=1, lines=lines)
        task = TranslationTask(chunk_index=1, chunk=chunk)
        
        # Create a mock result
        result = TranslationResult(
            chunk_index=1,
            translations={"1": "你好", "2": "世界"},
            missing_keys=[],
            unexpected_keys=[],
            processing_time=0.5,
            retry_count=0,
        )
        
        # Verify task and result properties
        self.assertEqual(task.chunk_index, 1)
        self.assertEqual(len(task.chunk.lines), 2)
        self.assertEqual(result.translations["1"], "你好")
        self.assertEqual(result.translations["2"], "世界")
    
    def test_rate_limiter_functionality(self):
        """Test rate limiter basic functionality"""
        limiter = RateLimiter(rate_per_minute=120)  # 2 per second
        
        async def test_rate_limit():
            start_time = time.time()
            
            # First acquisition should be immediate
            await limiter.acquire()
            first_time = time.time() - start_time
            self.assertLess(first_time, 0.1)
            
            # Second acquisition should also be immediate (2 per second allowed)
            await limiter.acquire()
            second_time = time.time() - start_time
            self.assertLess(second_time, 0.1)
            
            # Third acquisition might need to wait
            await limiter.acquire()
            third_time = time.time() - start_time
            # Should not take too long since we have 2 tokens per second
            self.assertLess(third_time, 1.0)
        
        asyncio.run(test_rate_limit())
    
    def test_concurrent_manager_initialization(self):
        """Test concurrent manager initialization"""
        # Test with different parameters
        manager1 = ConcurrentTranslationManager(max_concurrency=5)
        self.assertEqual(manager1.max_concurrency, 5)
        
        manager2 = ConcurrentTranslationManager(
            max_concurrency=2,
            rate_limit_per_minute=30,
            enable_circuit_breaker=False
        )
        self.assertEqual(manager2.max_concurrency, 2)
        self.assertEqual(manager2.rate_limit_per_minute, 30)
    
    def test_empty_document_handling(self):
        """Test handling of empty document"""
        empty_doc = SubtitleDocument(lines=[])
        
        config = LLMConfig(
            model="gpt-4o-mini",
            target_language="zh",
            batch_size=5,
            api_key_env="TEST_API_KEY",
            max_retries=3,
        )
        
        pipeline = PipelineConfig(
            prompt_preamble="Translate to $targetLanguage",
        )
        
        async def test_empty():
            # Should handle empty document without errors
            result = await self.manager.translate_document_concurrent(
                empty_doc, config, pipeline
            )
            self.assertEqual(len(result.lines), 0)
        
        asyncio.run(test_empty())
    
    def test_payload_building(self):
        """Test payload building for API requests"""
        lines = [
            SubtitleLine(index=1, start=0.0, end=1.0, text="Hello World"),
        ]
        chunk = TranslationChunk(index=1, lines=lines)
        
        config = LLMConfig(
            model="gpt-4o-mini",
            target_language="zh",
            temperature=0.2,
            api_key_env="TEST_API_KEY",
        )
        
        pipeline = PipelineConfig(
            prompt_preamble="Translate to $targetLanguage",
        )
        
        # Test payload building
        payload = self.manager._build_translation_payload(chunk, config, pipeline)
        
        # Verify payload structure
        self.assertEqual(payload["model"], "gpt-4o-mini")
        self.assertEqual(payload["temperature"], 0.2)
        self.assertIn("messages", payload)
        
        messages = payload["messages"]
        self.assertEqual(len(messages), 2)  # System and user messages
        
        # Check that user message contains the text to translate
        user_message = messages[1]["content"]
        self.assertIn("Hello World", user_message)
        self.assertIn("Target language: zh", user_message)
    
    def test_response_parsing(self):
        """Test parsing of translation API responses"""
        lines = [
            SubtitleLine(index=1, start=0.0, end=1.0, text="Hello"),
            SubtitleLine(index=2, start=1.0, end=2.0, text="World"),
        ]
        chunk = TranslationChunk(index=1, lines=lines)
        
        # Mock response
        mock_response = {
            "choices": [{
                "message": {
                    "content": '{"1": "你好", "2": "世界"}'
                }
            }]
        }
        
        # Test response parsing
        result = self.manager._parse_translation_response(mock_response, chunk)
        
        self.assertEqual(result.translations["1"], "你好")
        self.assertEqual(result.translations["2"], "世界")
        self.assertEqual(len(result.missing_keys), 0)
        self.assertEqual(len(result.unexpected_keys), 0)
    
    def test_response_parsing_with_markdown_fencing(self):
        """Test parsing responses with markdown fencing"""
        lines = [
            SubtitleLine(index=1, start=0.0, end=1.0, text="Hello"),
        ]
        chunk = TranslationChunk(index=1, lines=lines)
        
        # Mock response with markdown fencing
        mock_response = {
            "choices": [{
                "message": {
                    "content": '```json\n{"1": "你好"}\n```'
                }
            }]
        }
        
        result = self.manager._parse_translation_response(mock_response, chunk)
        self.assertEqual(result.translations["1"], "你好")
    
    def test_result_merging(self):
        """Test merging of translation results"""
        # Create mock results
        results = [
            TranslationResult(
                chunk_index=1,
                translations={"1": "A", "2": "B"},
                missing_keys=[],
                unexpected_keys=[],
                processing_time=0.1,
            ),
            TranslationResult(
                chunk_index=2,
                translations={"3": "C", "4": "D"},
                missing_keys=[],
                unexpected_keys=[],
                processing_time=0.1,
            ),
        ]
        
        existing = {"0": "Existing"}
        merged = self.manager._merge_results(results, existing)
        
        expected = {
            "0": "Existing",
            "1": "A",
            "2": "B",
            "3": "C",
            "4": "D",
        }
        self.assertEqual(merged, expected)
    
    def test_final_document_building(self):
        """Test building final translated document"""
        # Original document
        original_lines = [
            SubtitleLine(index=1, start=0.0, end=1.0, text="Hello"),
            SubtitleLine(index=2, start=1.0, end=2.0, text="World"),
        ]
        original_doc = SubtitleDocument(lines=original_lines)
        
        # Translations
        translations = {
            "1": "你好",
            "2": "世界",
        }
        
        # Build final document
        final_doc = self.manager._build_translated_document(original_doc, translations)
        
        # Verify final document
        self.assertEqual(len(final_doc.lines), 2)
        self.assertEqual(final_doc.lines[0].text, "你好")
        self.assertEqual(final_doc.lines[1].text, "世界")
        
        # Verify timing is preserved
        self.assertEqual(final_doc.lines[0].start, 0.0)
        self.assertEqual(final_doc.lines[0].end, 1.0)
        self.assertEqual(final_doc.lines[1].start, 1.0)
        self.assertEqual(final_doc.lines[1].end, 2.0)
    
    def test_statistics_tracking(self):
        """Test statistics collection"""
        # Simulate some operations
        self.manager.total_tasks = 10
        self.manager.completed_tasks = 8
        self.manager.failed_tasks = 2
        self.manager.total_processing_time = 4.0
        
        stats = self.manager.get_statistics()
        
        self.assertEqual(stats['total_tasks'], 10)
        self.assertEqual(stats['completed_tasks'], 8)
        self.assertEqual(stats['failed_tasks'], 2)
        self.assertEqual(stats['success_rate'], 0.8)
        self.assertEqual(stats['avg_processing_time'], 0.5)
    
    def test_error_response_handling(self):
        """Test handling of error responses"""
        lines = [
            SubtitleLine(index=1, start=0.0, end=1.0, text="Hello"),
        ]
        chunk = TranslationChunk(index=1, lines=lines)
        
        # Test with invalid JSON
        mock_response = {
            "choices": [{
                "message": {
                    "content": 'invalid json'
                }
            }]
        }
        
        with self.assertRaises(LLMTranslationError):
            self.manager._parse_translation_response(mock_response, chunk)
    
    def test_missing_translation_error(self):
        """Test error when translations are missing"""
        original_lines = [
            SubtitleLine(index=1, start=0.0, end=1.0, text="Hello"),
        ]
        original_doc = SubtitleDocument(lines=original_lines)
        
        # Empty translations
        translations = {}
        
        with self.assertRaises(LLMTranslationError) as context:
            self.manager._build_translated_document(original_doc, translations)
        
        self.assertIn("Missing translation for line index 1", str(context.exception))


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)