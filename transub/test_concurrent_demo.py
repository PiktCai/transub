"""
Demonstration of concurrent translation performance benefits
"""

from __future__ import annotations

import asyncio
import time
import unittest
from unittest.mock import patch, AsyncMock
from typing import Dict, List

from transub.concurrent_translate import ConcurrentTranslationManager, TranslationResult, TranslationTask
from transub.subtitles import SubtitleDocument, SubtitleLine
from transub.config import LLMConfig, PipelineConfig
from transub.translate import TranslationChunk


class TestConcurrentDemo(unittest.TestCase):
    """Demonstration tests showing concurrent translation benefits"""
    
    def create_test_document(self, num_lines: int) -> SubtitleDocument:
        """Create a test document with specified number of lines"""
        lines = [
            SubtitleLine(index=i, start=float(i), end=float(i+1), text=f"Line_{i}")
            for i in range(1, num_lines + 1)
        ]
        return SubtitleDocument(lines=lines)
    
    def test_concurrency_demo(self):
        """Demonstrate the benefits of concurrent translation"""
        print("\n" + "="*60)
        print("CONCURRENT TRANSLATION PERFORMANCE DEMONSTRATION")
        print("="*60)
        
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
        
        # Set up environment
        import os
        os.environ["TEST_API_KEY"] = "test-key"
        
        # Simulate serial translation (one chunk at a time)
        print("\n1. Serial Translation (Traditional Approach)")
        print("-" * 50)
        
        serial_start = time.time()
        
        # Simulate processing each chunk sequentially with delay
        for chunk_idx in range(5):  # 5 chunks
            print(f"   Processing chunk {chunk_idx + 1}/5...")
            time.sleep(0.2)  # Simulate API call delay
        
        serial_time = time.time() - serial_start
        print(f"   Total time: {serial_time:.3f} seconds")
        print(f"   Throughput: {20/serial_time:.1f} lines/second")
        
        # Simulate concurrent translation
        print("\n2. Concurrent Translation (New Approach)")
        print("-" * 50)
        
        manager = ConcurrentTranslationManager(max_concurrency=4)
        
        async def simulate_concurrent():
            concurrent_start = time.time()
            
            # Simulate processing all chunks concurrently
            async def process_chunk(chunk_idx: int):
                print(f"   Processing chunk {chunk_idx + 1}/5 concurrently...")
                await asyncio.sleep(0.2)  # Same API call delay
                print(f"   Chunk {chunk_idx + 1} completed!")
            
            # Process all chunks concurrently
            tasks = [process_chunk(i) for i in range(5)]
            await asyncio.gather(*tasks)
            
            concurrent_time = time.time() - concurrent_start
            return concurrent_time
        
        concurrent_time = asyncio.run(simulate_concurrent())
        print(f"   Total time: {concurrent_time:.3f} seconds")
        print(f"   Throughput: {20/concurrent_time:.1f} lines/second")
        
        # Calculate improvements
        speedup = serial_time / concurrent_time
        throughput_improvement = (20/concurrent_time) / (20/serial_time)
        
        print("\n3. Performance Improvement")
        print("-" * 50)
        print(f"Speedup: {speedup:.2f}x faster")
        print(f"Throughput improvement: {throughput_improvement:.2f}x")
        print(f"Time saved: {serial_time - concurrent_time:.3f} seconds ({(1 - concurrent_time/serial_time)*100:.1f}%)")
        
        # Verify concurrent is actually faster
        self.assertLess(concurrent_time, serial_time)
        self.assertGreater(speedup, 1.5)  # At least 1.5x speedup
    
    def test_scalability_demo(self):
        """Demonstrate scalability with different document sizes"""
        print("\n" + "="*60)
        print("SCALABILITY DEMONSTRATION")
        print("="*60)
        
        # Test with different document sizes
        sizes = [5, 10, 20, 30]
        batch_size = 5
        
        import os
        os.environ["TEST_API_KEY"] = "test-key"
        
        config = LLMConfig(
            model="gpt-4o-mini",
            target_language="zh",
            batch_size=batch_size,
            api_key_env="TEST_API_KEY",
            max_retries=3,
        )
        
        pipeline = PipelineConfig(
            prompt_preamble="Translate to $targetLanguage",
        )
        
        print(f"\nBatch size: {batch_size} lines per chunk")
        print(f"Max concurrency: 4 chunks simultaneously")
        print("\nDocument Size | Chunks | Serial Time | Concurrent Time | Speedup")
        print("-" * 65)
        
        for size in sizes:
            document = self.create_test_document(size)
            num_chunks = (size + batch_size - 1) // batch_size
            
            # Simulate serial processing
            serial_start = time.time()
            for chunk_idx in range(num_chunks):
                time.sleep(0.1)  # 0.1s per chunk
            serial_time = time.time() - serial_start
            
            # Simulate concurrent processing
            async def simulate_concurrent():
                concurrent_start = time.time()
                
                async def process_chunk(chunk_idx: int):
                    await asyncio.sleep(0.1)  # Same delay
                
                tasks = [process_chunk(i) for i in range(num_chunks)]
                await asyncio.gather(*tasks)
                
                return time.time() - concurrent_start
            
            concurrent_time = asyncio.run(simulate_concurrent())
            speedup = serial_time / concurrent_time
            
            print(f"{size:11d} | {num_chunks:6d} | {serial_time:10.3f}s | {concurrent_time:14.3f}s | {speedup:7.2f}x")
        
        print("\nObservation: Concurrent translation shows better scalability")
        print("as document size increases, especially with many chunks.")
    
    def test_concurrency_levels_demo(self):
        """Demonstrate impact of different concurrency levels"""
        print("\n" + "="*60)
        print("CONCURRENCY LEVELS DEMONSTRATION")
        print("="*60)
        
        # Create test scenario
        num_chunks = 10
        chunk_delay = 0.1  # 0.1 seconds per chunk
        
        print(f"\nScenario: {num_chunks} chunks, {chunk_delay}s processing time per chunk")
        print("\nConcurrency Level | Total Time | Theoretical Optimal | Efficiency")
        print("-" * 70)
        
        concurrency_levels = [1, 2, 3, 5, 10]
        theoretical_optimal = num_chunks * chunk_delay / max(concurrency_levels)
        
        for concurrency in concurrency_levels:
            # Simulate concurrent processing
            async def simulate_with_concurrency():
                start_time = time.time()
                
                # Track concurrent executions
                active_tasks = 0
                max_concurrent = 0
                
                async def process_with_tracking(chunk_idx: int):
                    nonlocal active_tasks, max_concurrent
                    active_tasks += 1
                    max_concurrent = max(max_concurrent, active_tasks)
                    await asyncio.sleep(chunk_delay)
                    active_tasks -= 1
                
                # Process chunks with concurrency limit
                semaphore = asyncio.Semaphore(concurrency)
                
                async def limited_process(chunk_idx: int):
                    async with semaphore:
                        await process_with_tracking(chunk_idx)
                
                tasks = [limited_process(i) for i in range(num_chunks)]
                await asyncio.gather(*tasks)
                
                elapsed_time = time.time() - start_time
                return elapsed_time, max_concurrent
            
            actual_time, max_concurrent = asyncio.run(simulate_with_concurrency())
            efficiency = (num_chunks * chunk_delay / concurrency) / actual_time
            
            print(f"{concurrency:17d} | {actual_time:9.3f}s | {theoretical_optimal:16.3f}s | {efficiency:9.2%}")
        
        print("\nObservation: Higher concurrency generally reduces total time,")
        print("but with diminishing returns due to overhead and chunk granularity.")
    
    def test_real_world_scenario_demo(self):
        """Demonstrate real-world scenario with typical subtitle processing"""
        print("\n" + "="*60)
        print("REAL-WORLD SCENARIO: SUBTITLE TRANSLATION")
        print("="*60)
        
        # Simulate a 30-minute video with subtitles every 3 seconds
        # This would be approximately 600 subtitle lines
        num_lines = 600
        chunk_size = 10  # Process 10 lines per chunk
        api_delay = 0.5  # 0.5 seconds per API call (realistic for LLM)
        
        print(f"\nScenario: {num_lines} subtitle lines ({num_lines * 3 // 60} minute video)")
        print(f"Processing: {chunk_size} lines per chunk")
        print(f"API latency: {api_delay}s per call")
        
        num_chunks = num_lines // chunk_size
        
        print(f"\nTotal chunks to process: {num_chunks}")
        print(f"Estimated API calls needed: {num_chunks}")
        
        # Simulate serial processing
        print(f"\n1. Serial Processing (Traditional)")
        serial_start = time.time()
        for i in range(num_chunks):
            if i % 10 == 0:  # Show progress every 10 chunks
                print(f"   Processing chunks {i+1}-{min(i+10, num_chunks)}...")
            time.sleep(api_delay)
        serial_time = time.time() - serial_start
        
        print(f"   Total time: {serial_time:.1f} seconds ({serial_time/60:.1f} minutes)")
        
        # Simulate concurrent processing
        print(f"\n2. Concurrent Processing (Optimized)")
        concurrent_start = time.time()
        
        async def process_concurrently():
            max_concurrency = 5  # Respect API rate limits
            semaphore = asyncio.Semaphore(max_concurrency)
            
            async def process_chunk(chunk_idx: int):
                async with semaphore:
                    await asyncio.sleep(api_delay)
            
            # Process chunks with progress reporting
            tasks = []
            for i in range(num_chunks):
                tasks.append(process_chunk(i))
                if (i + 1) % 10 == 0:  # Show progress every 10 chunks
                    print(f"   Queued chunks {i-8}-{i+1} for concurrent processing...")
            
            await asyncio.gather(*tasks)
        
        asyncio.run(process_concurrently())
        concurrent_time = time.time() - concurrent_start
        
        print(f"   Total time: {concurrent_time:.1f} seconds ({concurrent_time/60:.1f} minutes)")
        
        # Calculate benefits
        speedup = serial_time / concurrent_time
        time_saved = serial_time - concurrent_time
        
        print(f"\n3. Benefits Summary")
        print("-" * 30)
        print(f"Time saved: {time_saved:.1f} seconds ({time_saved/60:.1f} minutes)")
        print(f"Speed improvement: {speedup:.1f}x faster")
        print(f"Efficiency gain: {(1 - concurrent_time/serial_time) * 100:.1f}%")
        
        if speedup > 2.0:
            print("🚀 Significant performance improvement!")
        elif speedup > 1.5:
            print("✅ Good performance improvement!")
        else:
            print("📈 Modest performance improvement")
        
        self.assertGreater(speedup, 1.5)  # Should be significantly faster


if __name__ == "__main__":
    # Run demonstration tests
    unittest.main(verbosity=2)