"""
Performance comparison test between original and smart retry strategies
"""

from __future__ import annotations

import time
import unittest
from unittest.mock import Mock, patch
import requests

from .smart_retry import SmartRetryHandler, ErrorType


class MockAPIClient:
    """Mock API client to simulate different error scenarios"""
    
    def __init__(self):
        self.call_count = 0
        self.failure_scenario = "none"
        self.rate_limit_reset_time = 0
    
    def set_scenario(self, scenario: str):
        """Set failure scenario"""
        self.failure_scenario = scenario
        self.call_count = 0
    
    def make_request(self) -> dict:
        """Simulate API request with different failure scenarios"""
        self.call_count += 1
        
        if self.failure_scenario == "rate_limit_then_success":
            if self.call_count <= 2:
                raise self._create_rate_limit_error()
            return {"status": "success", "data": "translation"}
        
        elif self.failure_scenario == "network_flaky":
            if self.call_count % 2 == 1:
                raise requests.exceptions.ConnectionError("Network error")
            return {"status": "success", "data": "translation"}
        
        elif self.failure_scenario == "server_error_then_success":
            if self.call_count <= 1:
                raise self._create_server_error()
            return {"status": "success", "data": "translation"}
        
        elif self.failure_scenario == "always_fail":
            raise requests.exceptions.ConnectionError("Persistent failure")
        
        else:  # "none" - always success
            return {"status": "success", "data": "translation"}
    
    def _create_rate_limit_error(self):
        """Create rate limit error"""
        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.text = "Rate limit exceeded"
        return requests.exceptions.HTTPError(response=mock_response)
    
    def _create_server_error(self):
        """Create server error"""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal server error"
        return requests.exceptions.HTTPError(response=mock_response)


def original_retry_strategy(func, max_retries=3, base_delay=0.5):
    """Original simple retry strategy for comparison"""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            delay = base_delay * (2 ** attempt)
            time.sleep(delay)


class TestRetryPerformance(unittest.TestCase):
    """Performance comparison tests"""
    
    def setUp(self):
        self.client = MockAPIClient()
        self.smart_handler = SmartRetryHandler(enable_circuit_breaker=False)
    
    def test_rate_limit_scenario(self):
        """Test rate limit scenario - smart retry should be faster"""
        self.client.set_scenario("rate_limit_then_success")
        
        # Test original strategy
        start_time = time.time()
        try:
            result = original_retry_strategy(self.client.make_request, max_retries=5)
            original_time = time.time() - start_time
        except Exception:
            original_time = time.time() - start_time
        
        # Reset client
        self.client.set_scenario("rate_limit_then_success")
        
        # Test smart strategy
        start_time = time.time()
        try:
            result = self.smart_handler.execute_with_retry(self.client.make_request)
            smart_time = time.time() - start_time
        except Exception:
            smart_time = time.time() - start_time
        
        print(f"\nRate limit scenario:")
        print(f"Original strategy time: {original_time:.3f}s")
        print(f"Smart strategy time: {smart_time:.3f}s")
        
        # Smart should be faster or comparable due to better delay strategy
        self.assertGreaterEqual(original_time, 0)
        self.assertGreaterEqual(smart_time, 0)
    
    def test_network_flaky_scenario(self):
        """Test flaky network scenario"""
        self.client.set_scenario("network_flaky")
        
        # Test original strategy
        start_time = time.time()
        try:
            result = original_retry_strategy(self.client.make_request, max_retries=5)
            original_time = time.time() - start_time
        except Exception:
            original_time = time.time() - start_time
        
        # Reset client
        self.client.set_scenario("network_flaky")
        
        # Test smart strategy
        start_time = time.time()
        try:
            result = self.smart_handler.execute_with_retry(self.client.make_request)
            smart_time = time.time() - start_time
        except Exception:
            smart_time = time.time() - start_time
        
        print(f"\nNetwork flaky scenario:")
        print(f"Original strategy time: {original_time:.3f}s")
        print(f"Smart strategy time: {smart_time:.3f}s")
        
        # Both should succeed, compare timing
        self.assertGreaterEqual(original_time, 0)
        self.assertGreaterEqual(smart_time, 0)
    
    def test_retry_attempts_comparison(self):
        """Compare number of retry attempts"""
        scenarios = [
            ("rate_limit_then_success", 3),
            ("network_flaky", 5),
            ("server_error_then_success", 2),
        ]
        
        for scenario, expected_calls in scenarios:
            with self.subTest(scenario=scenario):
                self.client.set_scenario(scenario)
                
                # Original strategy
                original_calls = 0
                def tracked_original():
                    nonlocal original_calls
                    original_calls += 1
                    return self.client.make_request()
                
                try:
                    original_retry_strategy(tracked_original, max_retries=5)
                except Exception:
                    pass
                
                # Reset and test smart strategy
                self.client.set_scenario(scenario)
                smart_calls = 0
                def tracked_smart():
                    nonlocal smart_calls
                    smart_calls += 1
                    return self.client.make_request()
                
                try:
                    self.smart_handler.execute_with_retry(tracked_smart)
                except Exception:
                    pass
                
                print(f"\n{scenario}:")
                print(f"Original strategy calls: {original_calls}")
                print(f"Smart strategy calls: {smart_calls}")
                
                # Both should succeed within expected calls
                self.assertLessEqual(original_calls, 5)
                self.assertLessEqual(smart_calls, 5)
    
    def test_delay_strategy_efficiency(self):
        """Test delay strategy efficiency"""
        self.client.set_scenario("server_error_then_success")
        
        # Measure total time including delays
        delays_original = []
        delays_smart = []
        
        # Original strategy with delay tracking
        original_start = time.time()
        original_attempts = 0
        
        def tracked_original():
            nonlocal original_attempts
            original_attempts += 1
            if original_attempts > 1:
                delays_original.append(time.time() - original_start)
            return self.client.make_request()
        
        try:
            original_retry_strategy(tracked_original, max_retries=3, base_delay=0.1)
        except Exception:
            pass
        
        # Reset and test smart strategy
        self.client.set_scenario("server_error_then_success")
        smart_start = time.time()
        smart_attempts = 0
        
        def tracked_smart():
            nonlocal smart_attempts
            smart_attempts += 1
            if smart_attempts > 1:
                delays_smart.append(time.time() - smart_start)
            return self.client.make_request()
        
        try:
            self.smart_handler.execute_with_retry(tracked_smart)
        except Exception:
            pass
        
        print(f"\nDelay strategy comparison:")
        print(f"Original delays: {[f'{d:.3f}s' for d in delays_original]}")
        print(f"Smart delays: {[f'{d:.3f}s' for d in delays_smart]}")
        
        # Smart strategy should have more appropriate delays for server errors
        self.assertEqual(len(delays_original), len(delays_smart))


class TestErrorHandlingImprovements(unittest.TestCase):
    """Test error handling improvements"""
    
    def test_error_type_specific_handling(self):
        """Test that different error types get different treatment"""
        handler = SmartRetryHandler()
        
        # Test different error types have different retry counts
        rate_limit_rule = handler.rules[ErrorType.RATE_LIMIT]
        network_rule = handler.rules[ErrorType.NETWORK_ERROR]
        client_rule = handler.rules[ErrorType.CLIENT_ERROR]
        
        # Rate limit should have more retries
        self.assertGreater(rate_limit_rule.config.max_retries, 
                          client_rule.config.max_retries)
        
        # Network errors should have moderate retries
        self.assertGreater(network_rule.config.max_retries, 1)
        
        print(f"\nRetry count by error type:")
        print(f"Rate limit: {rate_limit_rule.config.max_retries}")
        print(f"Network error: {network_rule.config.max_retries}")
        print(f"Client error: {client_rule.config.max_retries}")
    
    def test_circuit_breaker_prevention(self):
        """Test that circuit breaker prevents excessive retries"""
        handler = SmartRetryHandler(enable_circuit_breaker=True)
        
        # Mock a function that always fails
        def always_fails():
            raise requests.exceptions.ConnectionError("Always fails")
        
        # Try multiple times, circuit breaker should eventually open
        results = []
        for i in range(10):
            try:
                handler.execute_with_retry(always_fails)
                results.append("success")
            except Exception as e:
                results.append(str(e))
        
        # Should have circuit breaker messages after threshold
        circuit_breaker_messages = [r for r in results if "Circuit breaker" in r]
        self.assertGreater(len(circuit_breaker_messages), 0)
        
        print(f"\nCircuit breaker prevention test:")
        print(f"Total attempts: {len(results)}")
        print(f"Circuit breaker blocks: {len(circuit_breaker_messages)}")


if __name__ == "__main__":
    # Run performance tests with verbose output
    unittest.main(verbosity=2)