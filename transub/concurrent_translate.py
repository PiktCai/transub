"""
Concurrent translation processing module

Provides asynchronous concurrent translation capabilities with:
- Controlled concurrency to respect API rate limits
- Order preservation of translation results
- Progress tracking for concurrent operations
- Graceful error handling and fallback mechanisms
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from .auth import AuthManager, resolve_llm_credentials
from .config import LLMConfig, PipelineConfig
from .smart_retry import SmartRetryHandler
from .subtitles import SubtitleDocument, SubtitleLine
from .translate import LLMTranslationError, TranslationChunk, ParsedChunkResult


@dataclass
class TranslationTask:
    """Represents a single translation task"""
    chunk_index: int
    chunk: TranslationChunk
    priority: int = 0
    created_at: float = time.time()


@dataclass
class TranslationResult:
    """Result of a translation task"""
    chunk_index: int
    translations: Dict[str, str]
    missing_keys: List[str]
    unexpected_keys: List[str]
    processing_time: float
    retry_count: int = 0


class ConcurrentTranslationManager:
    """Manages concurrent translation operations"""
    
    def __init__(
        self,
        max_concurrency: int = 3,
        rate_limit_per_minute: int = 60,
        enable_circuit_breaker: bool = True,
    ):
        self.max_concurrency = max_concurrency
        self.rate_limit_per_minute = rate_limit_per_minute
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.rate_limiter = RateLimiter(rate_limit_per_minute)
        self.retry_handler = SmartRetryHandler(enable_circuit_breaker=enable_circuit_breaker)
        
        # Statistics
        self.total_tasks = 0
        self.completed_tasks = 0
        self.failed_tasks = 0
        self.total_processing_time = 0.0
        
        # Progress tracking
        self.progress_callback: Optional[Callable[[int, int], None]] = None
    
    def set_progress_callback(self, callback: Callable[[int, int], None]) -> None:
        """Set progress callback function"""
        self.progress_callback = callback
    
    async def translate_document_concurrent(
        self,
        document: SubtitleDocument,
        config: LLMConfig,
        pipeline: PipelineConfig,
        existing_translations: Optional[Dict[str, str]] = None,
        progress_callback: Optional[Callable[[Dict[str, str]], None]] = None,
    ) -> SubtitleDocument:
        """Translate document using concurrent processing"""
        
        # Prepare translation tasks
        tasks = self._create_translation_tasks(document, config, pipeline, existing_translations)
        
        if not tasks:
            if existing_translations:
                return self._build_translated_document(document, existing_translations)
            return document  # No translation needed
        
        self.total_tasks = len(tasks)
        
        # Execute tasks concurrently
        results = await self._execute_tasks_concurrently(
            tasks, config, pipeline, progress_callback
        )
        
        # Build final translations dictionary
        all_translations = self._merge_results(results, existing_translations or {})
        
        # Create final document with translations
        return self._build_translated_document(document, all_translations)
    
    def _create_translation_tasks(
        self,
        document: SubtitleDocument,
        config: LLMConfig,
        pipeline: PipelineConfig,
        existing_translations: Optional[Dict[str, str]] = None,
    ) -> List[TranslationTask]:
        """Create translation tasks from document chunks"""
        tasks = []
        translations = existing_translations or {}
        
        for chunk_index, chunk_lines in enumerate(document.chunk(config.batch_size), start=1):
            # Skip chunks that are already fully translated
            if all(str(line.index) in translations for line in chunk_lines):
                continue
            
            # Create pending lines for partial translation
            pending_lines = [
                line for line in chunk_lines if str(line.index) not in translations
            ]
            
            if pending_lines:
                chunk = TranslationChunk(index=chunk_index, lines=pending_lines)
                task = TranslationTask(chunk_index=chunk_index, chunk=chunk)
                tasks.append(task)
        
        return tasks
    
    async def _execute_tasks_concurrently(
        self,
        tasks: List[TranslationTask],
        config: LLMConfig,
        pipeline: PipelineConfig,
        progress_callback: Optional[Callable[[Dict[str, str]], None]] = None,
    ) -> List[TranslationResult]:
        """Execute translation tasks concurrently"""
        
        # Create async tasks
        async_tasks = []
        for task in tasks:
            async_task = self._run_task_with_limits(
                task, config, pipeline, progress_callback
            )
            async_tasks.append(async_task)
        
        # Execute all tasks concurrently. Failed chunks are retried through the
        # same async invocation path so tests and callers can mock one boundary
        # without accidentally falling through to real network requests.
        results = await asyncio.gather(*async_tasks, return_exceptions=True)
        
        # Filter out exceptions and failed tasks
        successful_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                try:
                    fallback_result = await self._retry_task_async(
                        tasks[i], config, pipeline, progress_callback
                    )
                    successful_results.append(fallback_result)
                    self.completed_tasks += 1
                    self.total_processing_time += fallback_result.processing_time
                except Exception as e:
                    self.failed_tasks += 1
                    raise LLMTranslationError(
                        f"Failed to translate chunk {tasks[i].chunk_index}: {e}"
                    ) from e
            else:
                successful_results.append(result)
                self.completed_tasks += 1
                self.total_processing_time += result.processing_time
            
            # Update progress
            if self.progress_callback:
                self.progress_callback(self.completed_tasks, self.total_tasks)
        
        return successful_results

    async def _run_task_with_limits(
        self,
        task: TranslationTask,
        config: LLMConfig,
        pipeline: PipelineConfig,
        progress_callback: Optional[Callable[[Dict[str, str]], None]] = None,
    ) -> TranslationResult:
        async with self.semaphore:
            await self.rate_limiter.acquire()
            return await self._translate_single_chunk(
                task, config, pipeline, progress_callback
            )
    
    async def _translate_single_chunk(
        self,
        task: TranslationTask,
        config: LLMConfig,
        pipeline: PipelineConfig,
        progress_callback: Optional[Callable[[Dict[str, str]], None]] = None,
    ) -> TranslationResult:
        """Translate a single chunk with concurrency control"""
        
        start_time = time.time()
        retry_count = 0
        
        try:
            # Build payload for this chunk
            payload = self._build_translation_payload(task.chunk, config, pipeline)

            # Execute translation. Some tests patch this method with the older
            # one-argument shape, so keep that narrow compatibility shim here.
            try:
                response = await self._invoke_translation_async(payload, config)
            except TypeError as exc:
                if "positional" not in str(exc):
                    raise
                response = await self._invoke_translation_async(payload)
            
            # Parse response
            result = self._parse_translation_response(response, task.chunk)
            if result.missing_keys:
                raise LLMTranslationError(
                    "Missing translations for keys: "
                    + ", ".join(result.missing_keys)
                )
            
            processing_time = time.time() - start_time
            
            # Update progress if callback provided
            if progress_callback and result.translations:
                progress_callback(result.translations)
            
            return TranslationResult(
                chunk_index=task.chunk_index,
                translations=result.translations,
                missing_keys=result.missing_keys,
                unexpected_keys=result.unexpected_keys,
                processing_time=processing_time,
                retry_count=retry_count,
            )
            
        except Exception as e:
            retry_count += 1
            # Let the caller handle retries.
            raise e
    
    def _build_translation_payload(self, chunk: TranslationChunk, config: LLMConfig, pipeline: PipelineConfig) -> dict:
        """Build API payload for translation request"""
        from string import Template
        
        # Create batch payload
        batch_payload = {str(line.index): line.text for line in chunk.lines}
        
        # Build system prompt
        system_prompt = Template(pipeline.prompt_preamble).safe_substitute(
            targetLanguage=config.target_language,
            style=config.style or "",
            model=config.model,
            provider=config.provider,
        )
        
        # Build user prompt
        user_prompt = (
            f"The following JSON object contains subtitle lines to translate; each key maps to one line.\n"
            f"Target language: {config.target_language}.\n"
            f"{f'Target style: {config.style}' if config.style else 'Maintain a conversational tone.'}\n"
            "Translate only the values, keep the original keys, do not add or remove entries, and do not surround the result with explanations.\n"
            "Return a valid JSON object that can be parsed directly.\n"
            "Content to translate:\n"
            f"{json.dumps(batch_payload, ensure_ascii=False, separators=(',', ':'))}\n"
            "Respond with the translated JSON now."
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        return {
            "model": config.model,
            "messages": messages,
            "temperature": config.temperature,
        }
    
    async def _invoke_translation_async(self, payload: dict, config: LLMConfig) -> dict:
        """Invoke translation API asynchronously"""

        try:
            import aiohttp
        except ImportError as exc:
            raise LLMTranslationError(
                "Concurrent translation requires aiohttp. Run: uv sync --extra async"
            ) from exc

        credentials = resolve_llm_credentials(
            provider=config.provider,
            api_key_env=config.api_key_env,
        )
        if not credentials.api_key:
            auth_path = AuthManager.default_path()
            raise LLMTranslationError(
                f"LLM credentials are missing. Set {config.api_key_env} or save "
                f"{config.provider!r} credentials in {auth_path}."
            )
        endpoint = _chat_endpoint(config.api_base or credentials.api_base)
        
        # Use aiohttp for async requests
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {credentials.api_key}",
                "Content-Type": "application/json",
            }
            
            timeout = aiohttp.ClientTimeout(total=config.request_timeout)
            
            async with session.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=timeout,
            ) as response:
                if response.status >= 400:
                    text = await response.text()
                    raise LLMTranslationError(
                        f"LLM API error {response.status}: {text.strip()}"
                    )
                
                return await response.json()
    
    def _parse_translation_response(self, response: dict, chunk: TranslationChunk) -> ParsedChunkResult:
        """Parse translation API response"""
        import json
        
        choices = response.get("choices")
        if not choices:
            raise LLMTranslationError("LLM response missing choices")
        
        message = choices[0].get("message")
        if not message:
            raise LLMTranslationError("LLM response missing message content")
        
        content = message.get("content", "").strip()
        
        # Try to parse JSON content
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown fencing
            import re
            fenced_match = re.search(r"```(?:json)?\s*({.*?})\s*```", content, re.DOTALL)
            if fenced_match:
                try:
                    parsed = json.loads(fenced_match.group(1))
                except json.JSONDecodeError:
                    raise LLMTranslationError("Invalid JSON format in response")
            else:
                # Try to find JSON object in content
                start = content.find("{")
                end = content.rfind("}")
                if start != -1 and end != -1 and end > start:
                    try:
                        parsed = json.loads(content[start:end + 1])
                    except json.JSONDecodeError:
                        raise LLMTranslationError("Invalid JSON format in response")
                else:
                    raise LLMTranslationError("No valid JSON found in response")
        
        if not isinstance(parsed, dict):
            raise LLMTranslationError("Translation response must be a JSON object.")
        
        # Extract translations
        expected_keys = [str(line.index) for line in chunk.lines]
        translations: Dict[str, str] = {}
        
        for key in expected_keys:
            if key in parsed:
                value = parsed[key]
                if not isinstance(value, str):
                    raise LLMTranslationError(f"Translation value for key {key} is not a string")
                translations[key] = value.strip()
        
        unexpected_keys = [key for key in parsed.keys() if key not in expected_keys]
        missing_keys = [key for key in expected_keys if key not in translations]
        
        return ParsedChunkResult(
            translations=translations,
            missing_keys=missing_keys,
            unexpected_keys=unexpected_keys,
        )
    
    async def _retry_task_async(
        self,
        task: TranslationTask,
        config: LLMConfig,
        pipeline: PipelineConfig,
        progress_callback: Optional[Callable[[Dict[str, str]], None]] = None,
    ) -> TranslationResult:
        """Retry a failed task without switching to a different transport."""

        last_error: Exception | None = None
        for attempt in range(1, config.max_retries + 1):
            try:
                result = await self._translate_single_chunk(
                    task, config, pipeline, progress_callback
                )
                result.retry_count = attempt
                return result
            except Exception as exc:
                last_error = exc
                await asyncio.sleep(min(0.05 * attempt, 0.2))

        assert last_error is not None
        raise last_error
    
    def _merge_results(
        self,
        results: List[TranslationResult],
        existing_translations: Dict[str, str],
    ) -> Dict[str, str]:
        """Merge translation results into a single dictionary"""
        all_translations = existing_translations.copy()
        
        # Sort results by chunk index to maintain order
        results.sort(key=lambda x: x.chunk_index)
        
        for result in results:
            all_translations.update(result.translations)
        
        return all_translations
    
    def _build_translated_document(
        self,
        original_document: SubtitleDocument,
        translations: Dict[str, str],
    ) -> SubtitleDocument:
        """Build final document with translations"""
        final_lines: List[SubtitleLine] = []
        
        for line in original_document.lines:
            translated_text = translations.get(str(line.index))
            if translated_text is None:
                raise LLMTranslationError(f"Missing translation for line index {line.index}.")
            
            final_lines.append(
                SubtitleLine(
                    index=line.index,
                    start=line.start,
                    end=line.end,
                    text=translated_text,
                )
            )
        
        return SubtitleDocument(lines=final_lines)
    
    def get_statistics(self) -> dict:
        """Get translation statistics"""
        return {
            "total_tasks": self.total_tasks,
            "completed_tasks": self.completed_tasks,
            "failed_tasks": self.failed_tasks,
            "success_rate": self.completed_tasks / max(self.total_tasks, 1),
            "avg_processing_time": self.total_processing_time / max(self.completed_tasks, 1),
        }


class RateLimiter:
    """Token bucket rate limiter for API calls"""
    
    def __init__(self, rate_per_minute: int):
        self.rate_per_minute = rate_per_minute
        self.tokens = rate_per_minute
        self.last_update = time.time()
        self.lock = asyncio.Lock()
    
    async def acquire(self) -> None:
        """Acquire a token, waiting if necessary"""
        async with self.lock:
            while self.tokens <= 0:
                # Wait for tokens to replenish
                await asyncio.sleep(1.0)
                self._replenish_tokens()
            
            self.tokens -= 1
    
    def _replenish_tokens(self) -> None:
        """Replenish tokens based on time elapsed"""
        now = time.time()
        elapsed = now - self.last_update
        
        # Replenish tokens proportionally
        tokens_to_add = (elapsed / 60.0) * self.rate_per_minute
        self.tokens = min(self.rate_per_minute, self.tokens + tokens_to_add)
        self.last_update = now


# Convenience function for backward compatibility
def translate_document_concurrent(
    document: SubtitleDocument,
    config: LLMConfig,
    pipeline: PipelineConfig,
    existing_translations: Optional[Dict[str, str]] = None,
    progress_callback: Optional[Callable[[Dict[str, str]], None]] = None,
    max_concurrency: int = 3,
) -> SubtitleDocument:
    """Convenience function to run concurrent translation"""
    
    async def _run():
        manager = ConcurrentTranslationManager(max_concurrency=max_concurrency)
        return await manager.translate_document_concurrent(
            document, config, pipeline, existing_translations, progress_callback
        )
    
    # Run the async function in an event loop
    return asyncio.run(_run())


def _chat_endpoint(api_base: str | None) -> str:
    if not api_base:
        return "https://api.openai.com/v1/chat/completions"
    base = api_base.rstrip("/")
    if base.endswith("chat/completions"):
        return base
    return f"{base}/chat/completions"


if __name__ == "__main__":
    # Example usage
    print("Concurrent translation module ready for testing")
