"""
Smart retry strategy module

Provides intelligent retry mechanisms based on error types, including:
- Error type classification and identification
- Intelligent retry strategies
- Circuit breaker pattern
- Exponential backoff with random jitter
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, Optional, Type, Union

import requests


class ErrorType(Enum):
    """Error type enumeration"""
    RATE_LIMIT = "rate_limit"  # Rate limiting
    NETWORK_ERROR = "network_error"  # Network errors
    SERVER_ERROR = "server_error"  # Server errors
    TIMEOUT_ERROR = "timeout_error"  # Timeout errors
    CLIENT_ERROR = "client_error"  # Client errors
    UNKNOWN_ERROR = "unknown_error"  # Unknown errors


@dataclass
class RetryConfig:
    """Retry configuration"""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    jitter_range: tuple[float, float] = (0.5, 1.5)


class RetryRule:
    """Retry rule"""
    
    def __init__(
        self,
        error_type: ErrorType,
        max_retries: int,
        base_delay: float,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
    ):
        self.error_type = error_type
        self.config = RetryConfig(
            max_retries=max_retries,
            base_delay=base_delay,
            max_delay=max_delay,
            exponential_base=exponential_base,
            jitter=jitter,
        )


class CircuitBreaker:
    """Circuit breaker pattern implementation"""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: Type[Exception] = Exception,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.state = "closed"  # closed, open, half-open
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state == "open":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "half-open"
            else:
                raise Exception(f"Circuit breaker is open. Retry after {self.recovery_timeout} seconds.")
        
        try:
            result = func(*args, **kwargs)
            if self.state == "half-open":
                self.state = "closed"
                self.failure_count = 0
            return result
        except self.expected_exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
            raise e


class SmartRetryHandler:
    """Smart retry handler"""
    
    # Default retry rules
    DEFAULT_RULES = {
        ErrorType.RATE_LIMIT: RetryRule(
            error_type=ErrorType.RATE_LIMIT,
            max_retries=5,
            base_delay=2.0,
            max_delay=120.0,
        ),
        ErrorType.NETWORK_ERROR: RetryRule(
            error_type=ErrorType.NETWORK_ERROR,
            max_retries=3,
            base_delay=0.5,
            max_delay=30.0,
        ),
        ErrorType.SERVER_ERROR: RetryRule(
            error_type=ErrorType.SERVER_ERROR,
            max_retries=2,
            base_delay=2.0,
            max_delay=60.0,
        ),
        ErrorType.TIMEOUT_ERROR: RetryRule(
            error_type=ErrorType.TIMEOUT_ERROR,
            max_retries=3,
            base_delay=1.0,
            max_delay=45.0,
        ),
        ErrorType.CLIENT_ERROR: RetryRule(
            error_type=ErrorType.CLIENT_ERROR,
            max_retries=1,
            base_delay=1.0,
            max_delay=10.0,
        ),
        ErrorType.UNKNOWN_ERROR: RetryRule(
            error_type=ErrorType.UNKNOWN_ERROR,
            max_retries=2,
            base_delay=1.5,
            max_delay=30.0,
        ),
    }
    
    def __init__(
        self,
        rules: Optional[Dict[ErrorType, RetryRule]] = None,
        enable_circuit_breaker: bool = True,
    ):
        self.rules = rules or self.DEFAULT_RULES
        self.enable_circuit_breaker = enable_circuit_breaker
        
        if enable_circuit_breaker:
            self.circuit_breaker = CircuitBreaker(
                failure_threshold=5,
                recovery_timeout=60.0,
                expected_exception=Exception,
            )
        else:
            self.circuit_breaker = None
    
    def classify_error(self, exception: Exception) -> ErrorType:
        """Classify error type"""
        if isinstance(exception, requests.exceptions.HTTPError):
            if hasattr(exception, 'response') and exception.response is not None:
                status_code = exception.response.status_code
                if status_code == 429:
                    return ErrorType.RATE_LIMIT
                elif 500 <= status_code < 600:
                    return ErrorType.SERVER_ERROR
                elif 400 <= status_code < 500:
                    return ErrorType.CLIENT_ERROR
        
        if isinstance(exception, (requests.exceptions.ConnectionError, requests.exceptions.ConnectTimeout)):
            return ErrorType.NETWORK_ERROR
        
        if isinstance(exception, (requests.exceptions.Timeout, requests.exceptions.ReadTimeout)):
            return ErrorType.TIMEOUT_ERROR
        
        if isinstance(exception, requests.exceptions.RequestException):
            return ErrorType.NETWORK_ERROR
        
        return ErrorType.UNKNOWN_ERROR
    
    def calculate_delay(self, error_type: ErrorType, attempt: int) -> float:
        """Calculate retry delay"""
        rule = self.rules.get(error_type, self.rules[ErrorType.UNKNOWN_ERROR])
        config = rule.config
        
        # Exponential backoff
        delay = config.base_delay * (config.exponential_base ** (attempt - 1))
        
        # Cap maximum delay
        delay = min(delay, config.max_delay)
        
        # Add random jitter
        if config.jitter:
            jitter_factor = random.uniform(*config.jitter_range)
            delay *= jitter_factor
        
        return delay
    
    def execute_with_retry(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> any:
        """Execute function with intelligent retry mechanism"""
        
        def _execute():
            last_exception = None
            error_type = None
            
            # First attempt
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                error_type = self.classify_error(e)
            
            rule = self.rules.get(error_type, self.rules[ErrorType.UNKNOWN_ERROR])
            max_retries = rule.config.max_retries
            
            # Retry loop
            for attempt in range(1, max_retries + 1):
                delay = self.calculate_delay(error_type, attempt)
                
                # Log retry information
                print(f"Retry attempt {attempt}/{max_retries} for {error_type.value} error, "
                      f"waiting {delay:.2f} seconds...")
                
                time.sleep(delay)
                
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    # Re-classify error (error type might change)
                    error_type = self.classify_error(e)
            
            # All retries failed
            raise last_exception
        
        if self.circuit_breaker:
            return self.circuit_breaker.call(_execute)
        else:
            return _execute()


# Convenience function
def smart_retry(
    func: Callable,
    *args,
    max_retries: int = 3,
    base_delay: float = 1.0,
    **kwargs
) -> any:
    """Simple smart retry wrapper function"""
    handler = SmartRetryHandler()
    
    # Temporarily modify default rules
    for rule in handler.rules.values():
        rule.config.max_retries = max_retries
        rule.config.base_delay = base_delay
    
    return handler.execute_with_retry(func, *args, **kwargs)


if __name__ == "__main__":
    # Test code
    import requests
    
    def test_request():
        response = requests.get("https://httpbin.org/status/500")
        response.raise_for_status()
        return response.json()
    
    handler = SmartRetryHandler()
    
    try:
        result = handler.execute_with_retry(test_request)
        print("Success:", result)
    except Exception as e:
        print("Failed after all retries:", e)