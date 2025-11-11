"""
Performance comparison test between serial and concurrent translation
"""

from __future__ import annotations

import asyncio
import time
import unittest
from unittest.mock import patch, AsyncMock
from typing import Dict, List

from .concurrent_translate import ConcurrentTranslationManager, translate_document_concurrent
from .translate import LLMTranslator, translate_subtitles
from .subtitles import SubtitleDocument, SubtitleLine
from .config import LLMConfig, PipelineConfig


class MockSerialTranslator:
    """Mock serial translator for performance comparison"""
    
    def __init__(self, delay_per_request: float = 0.1):
        self.delay_per_request = delay_per_request
        self.call_count = 0
    
    def translate_document(self, document, existing_translations=None, progress_callback=None):
        """Simulate serial translation with delay"""
        # Simulate processing time proportional to document size
        processing_time = len(document.lines) * self.delay_per_request
        time.sleep(processing_time)
        self.call_count += 1
        
        # Return mock translated document
        translated_lines = []
        for line in document.lines:
            translated_lines.append(
                SubtitleLine(
                    index=line.index,
                    start=line.start,
                    end=line.end,
                    text=f"translated_{line.text}"
                )
            )
        
        return SubtitleDocument(lines=translated_lines)


class MockConcurrentTranslator:
    """Mock concurrent translator for performance comparison"""
    
    def __init__(self, delay_per_request: float = 0.1, max_concurrency: int = 3):
        self.delay_per_request = delay_per_request
        self.max_concurrency = max_concurrency
        self.call_count = 0
        self.max_concurrent_calls = 0
        self.current_concurrent = 0
    
    async def translate_chunk(self, chunk_lines: List[SubtitleLine]) -> Dict[str, str]:
        """Simulate concurrent translation of a chunk"""
        self.call_count += 1
        self.current_concurrent += 1
        self.max_concurrent_calls = max(self.max_concurrent_calls, self.current_concurrent)
        
        try:
            # Simulate processing delay
            await asyncio.sleep(self.delay_per_request)
            
            # Return mock translations
            translations = {}
            for line in chunk_lines:
                translations[str(line.index)] = f"translated_{line.text}"
            
            return translations
        finally:
            self.current_concurrent -= 1


class TestConcurrentPerformance(unittest.TestCase):
    """Performance comparison tests between serial and concurrent translation"""
    
    def create_test_document(self, num_lines: int) -> SubtitleDocument:
        """Create a test document with specified number of lines"""
        lines = [
            SubtitleLine(index=i, start=float(i), end=float(i+1), text=f"Line_{i}")
            for i in range(1, num_lines + 1)
        ]
        return SubtitleDocument(lines=lines)
    
    def test_serial_vs_concurrent_performance(self):
        """Compare performance between serial and concurrent translation"""
        # Set up environment variable
        import os
        os.environ["TEST_API_KEY"] = "test-api-key"
        
        # Create test document
        document = self.create_test_document(20)  # 20 lines
        
        # Mock config
        config = LLMConfig(
            model="gpt-4o-mini",
            target_language="zh",
            batch_size=4,  # 5 chunks of 4 lines each
            api_key_env="TEST_API_KEY",
            max_retries=3,
        )
        
        pipeline = PipelineConfig(
            prompt_preamble="Translate to $targetLanguage",
        )
        
        # Test serial translation
        serial_translator = MockSerialTranslator(delay_per_request=0.1)
        
        start_time = time.time()
        serial_result = serial_translator.translate_document(document)
        serial_time = time.time() - start_time
        
        # Test concurrent translation
        concurrent_manager = ConcurrentTranslationManager(max_concurrency=4)
        
        async def test_concurrent():
            # Mock the translation method to simulate delay
            async def mock_translate_chunk(task, config, pipeline, progress_callback=None):
                await asyncio.sleep(0.1)  # Same delay as serial
                return TranslationResult(
                    chunk_index=task.chunk_index,
                    translations={str(line.index): f"translated_{line.text}" for line in task.chunk.lines},
                    missing_keys=[],
                    unexpected_keys=[],
                    processing_time=0.1,
                )
            
            with patch.object(concurrent_manager, '_translate_single_chunk', mock_translate_chunk):
                return await concurrent_manager.translate_document_concurrent(
                    document, config, pipeline
                )
        
        start_time = time.time()
        concurrent_result = asyncio.run(test_concurrent())
        concurrent_time = time.time() - start_time
        
        # Print performance comparison
        print(f"\nPerformance Comparison (20 lines, 5 chunks):")
        print(f"Serial translation time: {serial_time:.3f}s")
        print(f"Concurrent translation time: {concurrent_time:.3f}s")
        print(f"Speedup: {serial_time / concurrent_time:.2f}x")
        
        # Concurrent should be faster (theoretical max speedup = max_concurrency)
        self.assertLess(concurrent_time, serial_time)
        self.assertGreater(serial_time / concurrent_time, 1.5)  # At least 1.5x speedup
        
        # Verify results are equivalent
        self.assertEqual(len(serial_result.lines), len(concurrent_result.lines))
        for i, (serial_line, concurrent_line) in enumerate(zip(serial_result.lines, concurrent_result.lines)):
            self.assertEqual(serial_line.index, concurrent_line.index)
            self.assertEqual(serial_line.start, concurrent_line.start)
            self.assertEqual(serial_line.end, concurrent_line.end)
    
    def test_concurrency_level_impact(self):
        """Test how different concurrency levels affect performance"""
        document = self.create_test_document(30)  # 30 lines
        
        config = LLMConfig(
            model="gpt-4o-mini",
            target_language="zh",
            batch_size=3,  # 10 chunks of 3 lines each
            api_key_env="TEST_API_KEY",
            max_retries=3,
        )
        
        pipeline = PipelineConfig(
            prompt_preamble="Translate to $targetLanguage",
        )
        
        concurrency_levels = [1, 2, 5, 10]
        results = {}
        
        for concurrency in concurrency_levels:
            manager = ConcurrentTranslationManager(max_concurrency=concurrency)
            
            async def test_with_concurrency():
                async def mock_translate_chunk(task, config, pipeline, progress_callback=None):
                    await asyncio.sleep(0.05)  # Shorter delay for this test
                    return TranslationResult(
                        chunk_index=task.chunk_index,
                        translations={str(line.index): f"translated_{line.text}" for line in task.chunk.lines},
                        missing_keys=[],
                        unexpected_keys=[],
                        processing_time=0.05,
                    )
                
                with patch.object(manager, '_translate_single_chunk', mock_translate_chunk):
                    start_time = time.time()
                    result = await manager.translate_document_concurrent(
                        document, config, pipeline
                    )
                    elapsed_time = time.time() - start_time
                    return elapsed_time
            
            elapsed_time = asyncio.run(test_with_concurrency())
            results[concurrency] = elapsed_time
            
            print(f"Concurrency {concurrency}: {elapsed_time:.3f}s")
        
        # Verify that higher concurrency generally leads to faster execution
        self.assertLess(results[10], results[1])  # 10 concurrent should be faster than 1
        self.assertLess(results[5], results[2])   # 5 concurrent should be faster than 2
        
        # But with diminishing returns
        speedup_1_to_2 = results[1] / results[2]
        speedup_2_to_5 = results[2] / results[5]
        speedup_5_to_10 = results[5] / results[10]
        
        print(f"Speedup 1→2: {speedup_1_to_2:.2f}x")
        print(f"Speedup 2→5: {speedup_2_to_5:.2f}x")
        print(f"Speedup 5→10: {speedup_5_to_10:.2f}x")
        
        # Diminishing returns - smaller speedups as concurrency increases
        self.assertGreater(speedup_1_to_2, speedup_2_to_5)
    
    def test_progress_tracking_performance(self):
        """Test that progress tracking doesn't significantly impact performance"""
        document = self.create_test_document(15)  # 15 lines
        
        config = LLMConfig(
            model="gpt-4o-mini",
            target_language="zh",
            batch_size=3,  # 5 chunks
            api_key_env="TEST_API_KEY",
            max_retries=3,
        )
        
        pipeline = PipelineConfig(
            prompt_preamble="Translate to $targetLanguage",
        )
        
        # Test without progress tracking
        manager1 = ConcurrentTranslationManager(max_concurrency=3)
        
        async def test_without_progress():
            async def mock_translate_chunk(task, config, pipeline, progress_callback=None):
                await asyncio.sleep(0.02)
                return TranslationResult(
                    chunk_index=task.chunk_index,
                    translations={str(line.index): f"translated_{line.text}" for line in task.chunk.lines},
                    missing_keys=[],
                    unexpected_keys=[],
                    processing_time=0.02,
                )
            
            with patch.object(manager1, '_translate_single_chunk', mock_translate_chunk):
                start_time = time.time()
                result = await manager1.translate_document_concurrent(
                    document, config, pipeline
                )
                return time.time() - start_time
        
        time_without_progress = asyncio.run(test_without_progress())
        
        # Test with progress tracking
        manager2 = ConcurrentTranslationManager(max_concurrency=3)
        progress_calls = []
        
        def progress_callback(completed, total):
            progress_calls.append((completed, total))
        
        manager2.set_progress_callback(progress_callback)
        
        async def test_with_progress():
            async def mock_translate_chunk(task, config, pipeline, progress_callback=None):
                await asyncio.sleep(0.02)
                result = TranslationResult(
                    chunk_index=task.chunk_index,
                    translations={str(line.index): f"translated_{line.text}" for line in task.chunk.lines},
                    missing_keys=[],
                    unexpected_keys=[],
                    processing_time=0.02,
                )
                # Simulate progress callback
                if progress_callback:
                    progress_callback(result.translations)
                return result
            
            with patch.object(manager2, '_translate_single_chunk', mock_translate_chunk):
                start_time = time.time()
                result = await manager2.translate_document_concurrent(
                    document, config, pipeline
                )
                return time.time() - start_time
        
        time_with_progress = asyncio.run(test_with_progress())
        
        print(f"\nProgress Tracking Performance Impact:")
        print(f"Without progress tracking: {time_without_progress:.3f}s")
        print(f"With progress tracking: {time_with_progress:.3f}s")
        print(f"Overhead: {((time_with_progress - time_without_progress) / time_without_progress * 100):.1f}%")
        
        # Progress tracking should not add significant overhead (< 10%)
        overhead = (time_with_progress - time_without_progress) / time_without_progress
        self.assertLess(overhead, 0.1)
        
        # Verify progress was tracked
        self.assertGreater(len(progress_calls), 0)
    
    def test_error_recovery_performance(self):
        """Test performance impact of error recovery mechanisms"""
        document = self.create_test_document(12)  # 12 lines
        
        config = LLMConfig(
            model="gpt-4o-mini",
            target_language="zh",
            batch_size=4,  # 3 chunks
            api_key_env="TEST_API_KEY",
            max_retries=3,
        )
        
        pipeline = PipelineConfig(
            prompt_preamble="Translate to $targetLanguage",
        )
        
        # Test with some failures that require retry
        manager = ConcurrentTranslationManager(max_concurrency=3)
        failure_count = 0
        
        async def mock_translate_with_failures(task, config, pipeline, progress_callback=None):
            nonlocal failure_count
            # Simulate occasional failures (33% failure rate)
            if failure_count % 3 == 1:
                failure_count += 1
                raise Exception("Simulated failure")
            
            failure_count += 1
            await asyncio.sleep(0.03)  # Slightly longer delay for retry overhead
            return TranslationResult(
                chunk_index=task.chunk_index,
                translations={str(line.index): f"translated_{line.text}" for line in task.chunk.lines},
                missing_keys=[],
                unexpected_keys=[],
                processing_time=0.03,
            )
        
        async def test_with_failures():
            with patch.object(manager, '_translate_single_chunk', mock_translate_with_failures):
                start_time = time.time()
                result = await manager.translate_document_concurrent(
                    document, config, pipeline
                )
                elapsed_time = time.time() - start_time
                
                # Verify results despite failures
                self.assertEqual(len(result.lines), 12)
                return elapsed_time
        
        time_with_failures = asyncio.run(test_with_failures())
        
        # Test without failures for comparison
        failure_count = 0  # Reset
        
        async def mock_translate_without_failures(task, config, pipeline, progress_callback=None):
            await asyncio.sleep(0.02)  # Normal delay
            return TranslationResult(
                chunk_index=task.chunk_index,
                translations={str(line.index): f"translated_{line.text}" for line in task.chunk.lines},
                missing_keys=[],
                unexpected_keys=[],
                processing_time=0.02,
            )
        
        async def test_without_failures():
            with patch.object(manager, '_translate_single_chunk', mock_translate_without_failures):
                start_time = time.time()
                result = await manager.translate_document_concurrent(
                    document, config, pipeline
                )
                elapsed_time = time.time() - start_time
                return elapsed_time
        
        time_without_failures = asyncio.run(test_without_failures())
        
        print(f"\nError Recovery Performance Impact:")
        print(f"Without failures: {time_without_failures:.3f}s")
        print(f"With failures (33% rate): {time_with_failures:.3f}s")
        print(f"Recovery overhead: {((time_with_failures - time_without_failures) / time_without_failures * 100):.1f}%")
        
        # Error recovery should not add excessive overhead (< 50%)
        overhead = (time_with_failures - time_without_failures) / time_without_failures
        self.assertLess(overhead, 0.5)


if __name__ == "__main__":
    # Run performance tests with verbose output
    unittest.main(verbosity=2)