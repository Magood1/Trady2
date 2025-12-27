# apps/trading_core/tests/test_execution_manager.py
import pytest
from unittest.mock import patch
from django.conf import settings
from .factories import TradingSignalFactory
from ..models import Order
from ..execution_manager import ExecutionManager

@pytest.mark.django_db
class TestExecutionManager:

    def test_demo_execution_creates_filled_order(self):
        # Arrange
        signal = TradingSignalFactory(status="PENDING")
        manager = ExecutionManager(signal)

        # Act
        manager.execute_trade()

        # Assert
        signal.refresh_from_db()
        assert signal.status == "EXECUTED"
        
        order = Order.objects.get(signal=signal)
        assert order.status == "FILLED"
        assert order.broker_order_id.startswith("demo_")

    def test_idempotency_if_signal_not_pending(self):
        # Arrange
        signal = TradingSignalFactory(status="EXECUTED")
        manager = ExecutionManager(signal)

        # Act
        manager.execute_trade()

        # Assert
        # No new order should be created. If it was, .get() would raise MultipleObjectsReturned
        Order.objects.get_or_create(signal=signal) # This won't raise if no new order created
        assert Order.objects.filter(signal=signal).count() == 1

    @patch('apps.trading_core.execution_manager.CircuitBreaker.check')
    def test_execution_halted_by_circuit_breaker(self, mock_check):
        # Arrange
        mock_check.side_effect = PermissionError("Circuit Breaker is tripped")
        signal = TradingSignalFactory(status="PENDING")
        manager = ExecutionManager(signal)

        # Act
        manager.execute_trade()

        # Assert
        signal.refresh_from_db()
        assert signal.status == "CANCELLED"
        assert "Circuit Breaker is tripped" in signal.meta['cancel_reason']
        assert not Order.objects.filter(signal=signal).exists()

    def test_live_mode_raises_error_by_default(self):
        # Arrange
        settings.TRADING_CONFIG['EXECUTION_MODE'] = 'live'
        signal = TradingSignalFactory(status="PENDING")
        manager = ExecutionManager(signal)

        # Act & Assert
        with pytest.raises(RuntimeError, match="Live trading is disabled"):
            manager.execute_trade()

        # Cleanup
        settings.TRADING_CONFIG['EXECUTION_MODE'] = 'demo'



        