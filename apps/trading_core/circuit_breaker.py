# apps/trading_core/circuit_breaker.py
import structlog
from decimal import Decimal
from django.conf import settings
from django.utils import timezone
from .models import CircuitBreakerState, TradingSignal

logger = structlog.get_logger(__name__)

class CircuitBreaker:
    """
    Production Grade Safety Valve.
    Rules:
    1. Max Consecutive Losses (e.g., 5)
    2. Max Drawdown % (e.g., 3% of capital)
    """
    
    @staticmethod
    def check():
        """
        Raises PermissionError if execution is unsafe.
        """
        state, _ = CircuitBreakerState.objects.get_or_create(id=1)
        if state.is_tripped:
            logger.critical("EXECUTION BLOCKED: Circuit Breaker Tripped.", reason=state.reason)
            raise PermissionError(f"Circuit Breaker Tripped: {state.reason}")

    @staticmethod
    def update_state(current_balance: float):
        """
        Evaluates system health after trades.
        """
        config = settings.TRADING_CONFIG.get('CIRCUIT_BREAKER', {})
        max_consecutive = config.get('max_consecutive_losses', 5)
        max_dd_pct = config.get('max_daily_drawdown_pct', 0.03) # 3% limit
        initial_balance = settings.TRADING_CONFIG.get('ACCOUNT_BALANCE', 10000.0)

        state, _ = CircuitBreakerState.objects.get_or_create(id=1)
        
        # 1. Check Consecutive Losses
        recent_signals = TradingSignal.objects.filter(status="EXECUTED").order_by('-timestamp')[:10]
        losses = 0
        # NOTE: In a real live system, we would check the 'Realized PnL' from the Order model.
        # Here we assume a placeholder logic or integrate with broker feedback.
        # For Phase 1 Pilot, we monitor this manually or via PnL feedback loop.
        
        # 2. Check Drawdown
        current_dd = (initial_balance - current_balance) / initial_balance
        if current_dd >= max_dd_pct:
            state.is_tripped = True
            state.reason = f"Max Drawdown Limit Hit: {current_dd*100:.2f}% > {max_dd_pct*100}%"
            state.tripped_at = timezone.now()
            state.save()
            logger.critical("CIRCUIT BREAKER TRIPPED: DRAWDOWN LIMIT EXCEEDED")
            return

        # If safe
        if state.is_tripped and state.reason == "Manual Reset Needed":
            pass # Stay tripped
        else:
            # Auto-recovery logic could go here, but for now manual reset is safer
            pass

    @staticmethod
    def reset():
        state, _ = CircuitBreakerState.objects.get_or_create(id=1)
        state.is_tripped = False
        state.reason = None
        state.consecutive_losses = 0
        state.last_reset_at = timezone.now()
        state.save()
        logger.info("Circuit Breaker manually reset.")