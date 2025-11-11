"""
Test module for smart retry functionality
"""

from __future__ import annotations

import json
import time
import unittest
from unittest.mock import Mock, patch, MagicMock
import requests

from .smart_retry import (
    SmartRetryHandler,
    ErrorType,
    RetryRule,
    CircuitBreaker,
    smart_retry,
)


class TestErrorClassification(unittest.TestCase):
    """Test error classification functionality"""
    
    def setUp(self):
        self.handler = SmartRetryHandler()
    
    def test_rate_limit_error(self):
        """Test rate limit error classification"""
        # Create a mock response with 429 status
        mock_response = Mock()
        mock_response.status_code = 429
        
        error = requests.exceptions.HTTPError(response=mock_response)
        error_type = self.handler.classify_error(error)
        
        self.assertEqual(error_type, ErrorType.RATE_LIMIT)
    
    def test_server_error(self):
        """Test server error classification"""
        # Test 500-level errors
        for status in [500, 502, 503, 504]:
            mock_response = Mock()
            mock_response.status_code = status
            
            error = requests.exceptions.HTTPError(response=mock_response)
            error_type = self.handler.classify_error(error)
            
            self.assertEqual(error_type, ErrorType.SERVER_ERROR)
    
    def test_network_error(self):
        """Test network error classification"""
        network_errors = [
            requests.exceptions.ConnectionError(),
            requests.exceptions.ConnectTimeout(),
        ]
        
        for error in network_errors:
            error_type = self.handler.classify_error(error)
            self.assertEqual(error_type, ErrorType.NETWORK_ERROR)
    
    def test_timeout_error(self):
        """Test timeout error classification"""
        timeout_errors = [
            requests.exceptions.Timeout(),
            requests.exceptions.ReadTimeout(),
        ]
        
        for error in timeout_errors:
            error_type = self.handler.classify_error(error)
            self.assertEqual(error_type, ErrorType.TIMEOUT_ERROR)


class TestRetryDelayCalculation(unittest.TestCase):
    """Test retry delay calculation"""
    
    def setUp(self):
        self.handler = SmartRetryHandler()
    
    def test_exponential_backoff(self):
        """Test exponential backoff calculation"""
        error_type = ErrorType.NETWORK_ERROR
        
        delay1 = self.handler.calculate_delay(error_type, 1)
        delay2 = self.handler.calculate_delay(error_type, 2)
        delay3 = self.handler.calculate_delay(error_type, 3)
        
        # Should be exponential growth (with jitter)
        self.assertLess(delay1, delay2)
        self.assertLess(delay2, delay3)
    
    def test_max_delay_cap(self):
        """Test maximum delay cap"""
        error_type = ErrorType.NETWORK_ERROR
        
        # Calculate delay for high attempt number
        delay = self.handler.calculate_delay(error_type, 8)  # Use reasonable attempt number
        rule = self.handler.rules[error_type]
        
        # Should not exceed max_delay (allow some tolerance for jitter)
        self.assertLessEqual(delay, rule.config.max_delay * 1.5)  # Allow 50% tolerance for jitter
    
    def test_jitter_effect(self):
        """Test jitter randomization effect"""
        error_type = ErrorType.NETWORK_ERROR
        
        # Calculate multiple delays for same parameters
        delays = [self.handler.calculate_delay(error_type, 2) for _ in range(10)]
        
        # Should have some variation due to jitter
        unique_delays = set(delays)
        self.assertGreater(len(unique_delays), 1)


class TestCircuitBreaker(unittest.TestCase):
    """Test circuit breaker functionality"""
    
    def setUp(self):
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=1.0,
        )
    
    def test_circuit_closed_initially(self):
        """Test circuit is closed initially"""
        def success_func():
            return "success"
        
        result = self.circuit_breaker.call(success_func)
        self.assertEqual(result, "success")
    
    def test_circuit_opens_after_failures(self):
        """Test circuit opens after threshold failures"""
        def failing_func():
            raise Exception("Test failure")
        
        # Fail threshold times
        for _ in range(3):
            with self.assertRaises(Exception):
                self.circuit_breaker.call(failing_func)
        
        # Circuit should be open now
        with self.assertRaises(Exception) as context:
            self.circuit_breaker.call(failing_func)
        
        self.assertIn("Circuit breaker is open", str(context.exception))
    
    def test_circuit_recovery(self):
        """Test circuit recovery after timeout"""
        def failing_func():
            raise Exception("Test failure")
        
        # Open the circuit
        for _ in range(3):
            try:
                self.circuit_breaker.call(failing_func)
            except Exception:
                pass
        
        # Wait for recovery timeout
        time.sleep(1.1)
        
        # Circuit should be half-open, then closed on success
        def success_func():
            return "recovered"
        
        result = self.circuit_breaker.call(success_func)
        self.assertEqual(result, "recovered")
        self.assertEqual(self.circuit_breaker.state, "closed")


class TestSmartRetryIntegration(unittest.TestCase):
    """Test smart retry integration with mock functions"""
    
    def setUp(self):
        self.handler = SmartRetryHandler(enable_circuit_breaker=False)
    
    def test_successful_function_execution(self):
        """Test successful function execution"""
        def success_func():
            return {"status": "success"}
        
        result = self.handler.execute_with_retry(success_func)
        self.assertEqual(result, {"status": "success"})
    
    def test_retry_on_transient_failure(self):
        """Test retry on transient failure"""
        call_count = 0
        
        def transient_failure_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise requests.exceptions.ConnectionError("Network error")
            return {"status": "success"}
        
        result = self.handler.execute_with_retry(transient_failure_func)
        self.assertEqual(result, {"status": "success"})
        self.assertEqual(call_count, 3)
    
    def test_final_failure_after_retries(self):
        """Test final failure after all retries exhausted"""
        def always_failing_func():
            raise requests.exceptions.ConnectionError("Persistent network error")
        
        with self.assertRaises(requests.exceptions.ConnectionError):
            self.handler.execute_with_retry(always_failing_func)


class TestRetryRules(unittest.TestCase):
    """Test retry rules configuration"""
    
    def test_custom_retry_rules(self):
        """Test custom retry rules"""
        custom_rules = {
            ErrorType.RATE_LIMIT: RetryRule(
                error_type=ErrorType.RATE_LIMIT,
                max_retries=10,
                base_delay=5.0,
                max_delay=300.0,
            ),
        }
        
        handler = SmartRetryHandler(rules=custom_rules)
        rule = handler.rules[ErrorType.RATE_LIMIT]
        
        self.assertEqual(rule.config.max_retries, 10)
        self.assertEqual(rule.config.base_delay, 5.0)
        self.assertEqual(rule.config.max_delay, 300.0)


class TestConvenienceFunction(unittest.TestCase):
    """Test convenience wrapper function"""
    
    def test_smart_retry_convenience_function(self):
        """Test smart_retry convenience function"""
        def success_func():
            return "convenience success"
        
        result = smart_retry(success_func, max_retries=5, base_delay=0.1)
        self.assertEqual(result, "convenience success")


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)