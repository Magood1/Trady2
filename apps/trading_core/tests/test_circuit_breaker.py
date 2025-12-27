# apps/trading_core/tests/test_circuit_breaker.py
import pytest
from django.conf import settings
from .factories import TradingSignalFactory
from ..circuit_breaker import CircuitBreaker
from ..models import CircuitBreakerState

@pytest.mark.django_db
class TestCircuitBreaker:
    
    def setup_method(self):
        CircuitBreaker.reset()

    def teardown_method(self):
        CircuitBreaker.reset()

    def test_check_passes_when_not_tripped(self):
        try:
            CircuitBreaker.check()
        except PermissionError:
            pytest.fail("CircuitBreaker.check() raised PermissionError unexpectedly.")

    def test_check_raises_error_when_tripped(self):
        state, _ = CircuitBreakerState.objects.get_or_create(id=1)
        state.is_tripped = True
        state.reason = "Test trip"
        state.save()

        with pytest.raises(PermissionError, match="Circuit Breaker is tripped: Test trip"):
            CircuitBreaker.check()

    # NOTE: The update_state logic is simplified. A more robust test would require
    # mocking the P&L of closed trades. This test validates the current simplified logic.
    def test_trips_on_consecutive_losses(self):
        # Arrange
        max_losses = settings.TRADING_CONFIG['CIRCUIT_BREAKER']['max_consecutive_losses']
        for _ in range(max_losses):
            TradingSignalFactory(status="EXECUTED") # Simplified: assume all are losses

        # Act
        CircuitBreaker.update_state()

        # Assert
        state, _ = CircuitBreakerState.objects.get_or_create(id=1)
        assert state.is_tripped is True


        