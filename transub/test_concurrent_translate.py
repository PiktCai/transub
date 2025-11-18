"""
Test module for concurrent translation functionality
"""

from __future__ import annotations

import asyncio
import time
import unittest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Dict, List

from transub.concurrent_translate import (
    ConcurrentTranslationManager,
    TranslationTask,
    TranslationResult,
    RateLimiter,
    translate_document_concurrent,
)
from transub.subtitles import SubtitleDocument, SubtitleLine
from transub.config import LLMConfig, PipelineConfig
from transub.translate import LLMTranslationError


class MockLLMClient:
    """Mock LLM client for testing concurrent operations"""
    
    def __init__(self):
        self.call_count = 0
        self.active_requests = 0
        self.max_concurrent = 0
        self.failure_rate = 0.0
        self.delay = 0.1
        self.request_log = []
    
    def set_failure_rate(self, rate: float):
        """Set failure rate for testing"""
        self.failure_rate = rate
    
    def set_delay(self, delay: float):
        """Set response delay for testing"""
        self.delay = delay
    
    async def make_request(self, payload: dict) -> dict:
        """Simulate async API request"""
        self.call_count += 1
        self.active_requests += 1
        self.max_concurrent = max(self.max_concurrent, self.active_requests)
        
        # Record request for analysis
        self.request_log.append({
            'timestamp': time.time(),
            'concurrent': self.active_requests,
            'payload': payload,
        })
        
        try:
            # Simulate processing delay
            await asyncio.sleep(self.delay)
            
            # Simulate failures
            if self.failure_rate > 0 and (self.call_count % int(1/self.failure_rate) == 0):
                raise Exception(f"Simulated failure for request {self.call_count}")
            
            # Generate mock response based on request
            messages = payload.get('messages', [])
            if messages and len(messages) > 1:
                user_content = messages[1].get('content', '')
                # Extract JSON from user content and return translated version
                import json
                try:
                    # Simple mock: reverse the text as "translation"
                    if 'Content to translate:' in user_content:
                        # Extract the JSON part from the content
                        lines = user_content.split('\n')
                        json_line = None
                        for line in lines:
                            line = line.strip()
                            if line.startswith('{') and line.endswith('}'):
                                json_line = line
                                break
                        
                        if json_line:
                            data = json.loads(json_line)
                            # Create translated response with same keys but reversed values
                            translated = {}
                            for k, v in data.items():
                                if isinstance(v, str):
                                    translated[k] = v[::-1]  # Reverse as "translation"
                                else:
                                    translated[k] = str(v)
                            
                            return {
                                "choices": [{
                                    "message": {
                                        "content": json.dumps(translated, ensure_ascii=False)
                                    }
                                }]
                            }
                except Exception as e:
                    print(f"Mock translation error: {e}")
                    pass
            
            # Default response - create based on request patterns
            # Extract indices from the request to create appropriate response
            import re
            content_str = str(payload)
            indices = re.findall(r'"(\d+)"', content_str)
            
            if indices:
                # Create response with the found indices
                response_data = {idx: f"translated_{idx}" for idx in indices[:5]}  # Limit to 5 items
                return {
                    "choices": [{
                        "message": {
                            "content": json.dumps(response_data)
                        }
                    }]
                }
            
            # Fallback default response
            return {
                "choices": [{
                    "message": {
                        "content": '{"1": "translated text"}'
                    }
                }]
            }
            
        finally:
            self.active_requests -= 1


class TestConcurrentTranslationManager(unittest.TestCase):
    """Test ConcurrentTranslationManager functionality"""
    
    def setUp(self):
        self.manager = ConcurrentTranslationManager(max_concurrency=3)
        self.mock_client = MockLLMClient()
    
    def test_basic_concurrent_translation(self):
        """Test basic concurrent translation functionality"""
        # Create test document
        lines = [
            SubtitleLine(index=i, start=0.0, end=1.0, text=f"Text {i}")
            for i in range(1, 11)  # 10 lines
        ]
        document = SubtitleDocument(lines=lines)
        
        # Mock config with environment variable
        import os
        os.environ["TEST_API_KEY"] = "test-api-key"
        
        config = LLMConfig(
            model="gpt-4o-mini",
            target_language="zh",
            batch_size=2,  # Small batches for testing
            api_key_env="TEST_API_KEY",  # Use environment variable
            max_retries=3,
        )
        
        pipeline = PipelineConfig(
            prompt_preamble="Translate to $targetLanguage",
        )
        
        async def run_test():
            # Patch the async invoke method properly
            with patch.object(self.manager, '_invoke_translation_async', return_value={
                "choices": [{
                    "message": {
                        "content": '{"1": "translated_1", "2": "translated_2"}'
                    }
                }]
            }) as mock_invoke:
                result = await self.manager.translate_document_concurrent(
                    document, config, pipeline
                )
                
                # Verify results
                self.assertIsInstance(result, SubtitleDocument)
                self.assertEqual(len(result.lines), 10)
                
                # Verify API was called for translation
                self.assertGreater(mock_invoke.call_count, 0)
                
                return result
        
        # Run async test
        result = asyncio.run(run_test())
        
        # Verify concurrent execution happened
        self.assertGreater(self.mock_client.max_concurrent, 1)
        self.assertLessEqual(self.mock_client.max_concurrent, 3)  # Should respect max_concurrency
    
    def test_concurrency_limit_enforcement(self):
        """Test that concurrency limits are properly enforced"""
        # Create test with many small chunks
        lines = [
            SubtitleLine(index=i, start=0.0, end=1.0, text=f"Text {i}")
            for i in range(1, 21)  # 20 lines
        ]
        document = SubtitleDocument(lines=lines)
        
        config = LLMConfig(
            model="gpt-4o-mini",
            target_language="zh",
            batch_size=1,  # One line per chunk to maximize concurrency
            api_key="test-key",
            max_retries=3,
        )
        
        pipeline = PipelineConfig(
            prompt_preamble="Translate to $targetLanguage",
        )
        
        async def run_test():
            with patch.object(self.manager, '_invoke_translation_async', self.mock_client.make_request):
                await self.manager.translate_document_concurrent(document, config, pipeline)
                
                # Verify concurrency was limited
                self.assertLessEqual(self.mock_client.max_concurrent, 3)
                self.assertGreater(self.mock_client.call_count, 15)  # Should process many requests
        
        asyncio.run(run_test())
    
    def test_order_preservation(self):
        """Test that translation results maintain original order"""
        # Create document with identifiable content
        lines = [
            SubtitleLine(index=i, start=float(i), end=float(i+1), text=f"Line {i:02d}")
            for i in range(1, 6)  # 5 lines
        ]
        document = SubtitleDocument(lines=lines)
        
        config = LLMConfig(
            model="gpt-4o-mini",
            target_language="zh",
            batch_size=1,
            api_key="test-key",
            max_retries=3,
        )
        
        pipeline = PipelineConfig(
            prompt_preamble="Translate to $targetLanguage",
        )
        
        async def run_test():
            with patch.object(self.manager, '_invoke_translation_async', self.mock_client.make_request):
                result = await self.manager.translate_document_concurrent(document, config, pipeline)
                
                # Verify order is preserved
                for i, line in enumerate(result.lines, 1):
                    self.assertEqual(line.index, i)
                    self.assertEqual(line.start, float(i))
                    self.assertEqual(line.end, float(i+1))
                    # Mock translation reverses the text
                    self.assertEqual(line.text, f"20 diL")  # "Line {i:02d}" reversed
        
        asyncio.run(run_test())
    
    def test_error_handling_and_fallback(self):
        """Test error handling and synchronous fallback"""
        # Set up mock to fail sometimes
        self.mock_client.set_failure_rate(0.3)  # 30% failure rate
        
        lines = [
            SubtitleLine(index=i, start=0.0, end=1.0, text=f"Text {i}")
            for i in range(1, 11)
        ]
        document = SubtitleDocument(lines=lines)
        
        config = LLMConfig(
            model="gpt-4o-mini",
            target_language="zh",
            batch_size=2,
            api_key="test-key",
            max_retries=3,
        )
        
        pipeline = PipelineConfig(
            prompt_preamble="Translate to $targetLanguage",
        )
        
        async def run_test():
            with patch.object(self.manager, '_invoke_translation_async', self.mock_client.make_request):
                # Should handle failures gracefully
                result = await self.manager.translate_document_concurrent(document, config, pipeline)
                
                # Should still get a result despite failures
                self.assertIsInstance(result, SubtitleDocument)
                self.assertEqual(len(result.lines), 10)
        
        asyncio.run(run_test())
    
    def test_progress_tracking(self):
        """Test progress tracking functionality"""
        progress_updates = []
        
        def progress_callback(completed: int, total: int):
            progress_updates.append((completed, total))
        
        self.manager.set_progress_callback(progress_callback)
        
        lines = [
            SubtitleLine(index=i, start=0.0, end=1.0, text=f"Text {i}")
            for i in range(1, 6)
        ]
        document = SubtitleDocument(lines=lines)
        
        config = LLMConfig(
            model="gpt-4o-mini",
            target_language="zh",
            batch_size=1,
            api_key="test-key",
            max_retries=3,
        )
        
        pipeline = PipelineConfig(
            prompt_preamble="Translate to $targetLanguage",
        )
        
        async def run_test():
            with patch.object(self.manager, '_invoke_translation_async', self.mock_client.make_request):
                await self.manager.translate_document_concurrent(document, config, pipeline)
                
                # Verify progress updates
                self.assertGreater(len(progress_updates), 0)
                
                # Final progress should show completion
                final_completed, final_total = progress_updates[-1]
                self.assertEqual(final_completed, final_total)
        
        asyncio.run(run_test())
    
    def test_statistics_collection(self):
        """Test statistics collection"""
        lines = [
            SubtitleLine(index=i, start=0.0, end=1.0, text=f"Text {i}")
            for i in range(1, 6)
        ]
        document = SubtitleDocument(lines=lines)
        
        config = LLMConfig(
            model="gpt-4o-mini",
            target_language="zh",
            batch_size=1,
            api_key="test-key",
            max_retries=3,
        )
        
        pipeline = PipelineConfig(
            prompt_preamble="Translate to $targetLanguage",
        )
        
        async def run_test():
            with patch.object(self.manager, '_invoke_translation_async', self.mock_client.make_request):
                await self.manager.translate_document_concurrent(document, config, pipeline)
                
                # Check statistics
                stats = self.manager.get_statistics()
                self.assertEqual(stats['total_tasks'], 5)
                self.assertEqual(stats['completed_tasks'], 5)
                self.assertEqual(stats['failed_tasks'], 0)
                self.assertEqual(stats['success_rate'], 1.0)
                self.assertGreater(stats['avg_processing_time'], 0)
        
        asyncio.run(run_test())


class TestRateLimiter(unittest.TestCase):
    """Test RateLimiter functionality"""
    
    def test_rate_limiting_basic(self):
        """Test basic rate limiting"""
        limiter = RateLimiter(rate_per_minute=60)  # 1 per second
        
        async def run_test():
            start_time = time.time()
            
            # Acquire 2 tokens with timing check
            await limiter.acquire()  # First should be immediate
            first_time = time.time() - start_time
            
            await limiter.acquire()  # Second should wait
            second_time = time.time() - start_time
            
            # Should take about 1 second for the second acquisition
            self.assertGreaterEqual(second_time, 0.9)  # Allow small margin
            self.assertGreaterEqual(second_time, first_time)
        
        asyncio.run(run_test())
    
    def test_rate_limiter_token_replenishment(self):
        """Test token replenishment over time"""
        limiter = RateLimiter(rate_per_minute=60)  # 1 per second
        
        async def run_test():
            # Use up tokens
            await limiter.acquire()
            self.assertEqual(limiter.tokens, 59)
            
            # Wait for replenishment
            await asyncio.sleep(1.1)
            limiter._replenish_tokens()
            self.assertGreaterEqual(limiter.tokens, 60)
        
        asyncio.run(run_test())


class TestTranslationTask(unittest.TestCase):
    """Test TranslationTask dataclass"""
    
    def test_task_creation(self):
        """Test task creation and attributes"""
        from .translate import TranslationChunk
        
        chunk = TranslationChunk(
            index=1,
            lines=[
                SubtitleLine(index=1, start=0.0, end=1.0, text="Test"),
                SubtitleLine(index=2, start=1.0, end=2.0, text="Text"),
            ]
        )
        
        task = TranslationTask(
            chunk_index=1,
            chunk=chunk,
            priority=5,
        )
        
        self.assertEqual(task.chunk_index, 1)
        self.assertEqual(task.chunk, chunk)
        self.assertEqual(task.priority, 5)
        self.assertIsInstance(task.created_at, float)


class TestTranslationResult(unittest.TestCase):
    """Test TranslationResult dataclass"""
    
    def test_result_creation(self):
        """Test result creation and attributes"""
        result = TranslationResult(
            chunk_index=1,
            translations={"1": "translated", "2": "text"},
            missing_keys=[],
            unexpected_keys=[],
            processing_time=1.5,
            retry_count=2,
        )
        
        self.assertEqual(result.chunk_index, 1)
        self.assertEqual(result.translations, {"1": "translated", "2": "text"})
        self.assertEqual(result.processing_time, 1.5)
        self.assertEqual(result.retry_count, 2)


class TestConvenienceFunction(unittest.TestCase):
    """Test convenience wrapper function"""
    
    def test_convenience_function(self):
        """Test the convenience wrapper function"""
        lines = [
            SubtitleLine(index=i, start=0.0, end=1.0, text=f"Text {i}")
            for i in range(1, 4)
        ]
        document = SubtitleDocument(lines=lines)
        
        config = LLMConfig(
            model="gpt-4o-mini",
            target_language="zh",
            batch_size=1,
            api_key="test-key",
            max_retries=3,
        )
        
        pipeline = PipelineConfig(
            prompt_preamble="Translate to $targetLanguage",
        )
        
        # Mock the async implementation
        with patch('transub.concurrent_translate.ConcurrentTranslationManager.translate_document_concurrent') as mock_translate:
            mock_translate.return_value = document  # Return same document for simplicity
            
            result = translate_document_concurrent(
                document, config, pipeline, max_concurrency=2
            )
            
            self.assertIsInstance(result, SubtitleDocument)
            mock_translate.assert_called_once()


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions"""
    
    def setUp(self):
        self.manager = ConcurrentTranslationManager()
        self.mock_client = MockLLMClient()
    
    def test_empty_document(self):
        """Test translation of empty document"""
        document = SubtitleDocument(lines=[])
        
        config = LLMConfig(
            model="gpt-4o-mini",
            target_language="zh",
            batch_size=5,
            api_key="test-key",
            max_retries=3,
        )
        
        pipeline = PipelineConfig(
            prompt_preamble="Translate to $targetLanguage",
        )
        
        async def run_test():
            with patch.object(self.manager, '_invoke_translation_async', self.mock_client.make_request):
                result = await self.manager.translate_document_concurrent(document, config, pipeline)
                
                self.assertEqual(len(result.lines), 0)
                self.assertEqual(self.manager.completed_tasks, 0)
        
        manager = ConcurrentTranslationManager()
        self.manager = manager
        asyncio.run(run_test())
    
    def test_all_lines_already_translated(self):
        """Test when all lines already have translations"""
        lines = [
            SubtitleLine(index=i, start=0.0, end=1.0, text=f"Text {i}")
            for i in range(1, 4)
        ]
        document = SubtitleDocument(lines=lines)
        
        # Provide existing translations
        existing_translations = {
            "1": "Translated 1",
            "2": "Translated 2",
            "3": "Translated 3",
        }
        
        config = LLMConfig(
            model="gpt-4o-mini",
            target_language="zh",
            batch_size=5,
            api_key="test-key",
            max_retries=3,
        )
        
        pipeline = PipelineConfig(
            prompt_preamble="Translate to $targetLanguage",
        )
        
        async def run_test():
            with patch.object(self.manager, '_invoke_translation_async', self.mock_client.make_request) as mock_invoke:
                result = await self.manager.translate_document_concurrent(
                    document, config, pipeline, existing_translations=existing_translations
                )
                
                # Should not make any API calls
                mock_invoke.assert_not_called()
                
                # Should return document with existing translations
                self.assertEqual(len(result.lines), 3)
                for line in result.lines:
                    self.assertEqual(line.text, existing_translations[str(line.index)])
        
        manager = ConcurrentTranslationManager()
        self.manager = manager
        asyncio.run(run_test())
    
    def test_partial_existing_translations(self):
        """Test when some lines already have translations"""
        lines = [
            SubtitleLine(index=i, start=0.0, end=1.0, text=f"Text {i}")
            for i in range(1, 6)
        ]
        document = SubtitleDocument(lines=lines)
        
        # Provide partial existing translations
        existing_translations = {
            "1": "Translated 1",
            "3": "Translated 3",
            "5": "Translated 5",
        }
        
        config = LLMConfig(
            model="gpt-4o-mini",
            target_language="zh",
            batch_size=5,  # All in one chunk
            api_key="test-key",
            max_retries=3,
        )
        
        pipeline = PipelineConfig(
            prompt_preamble="Translate to $targetLanguage",
        )
        
        async def run_test():
            with patch.object(self.manager, '_invoke_translation_async', self.mock_client.make_request):
                result = await self.manager.translate_document_concurrent(
                    document, config, pipeline, existing_translations=existing_translations
                )
                
                # Should complete successfully
                self.assertEqual(len(result.lines), 5)
                
                # Check that existing translations are preserved
                self.assertEqual(result.lines[0].text, "Translated 1")  # index 1
                self.assertEqual(result.lines[2].text, "Translated 3")  # index 3
                self.assertEqual(result.lines[4].text, "Translated 5")  # index 5
        
        manager = ConcurrentTranslationManager()
        self.manager = manager
        asyncio.run(run_test())


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)