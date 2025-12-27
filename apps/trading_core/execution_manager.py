# apps/trading_core/execution_manager.py
import structlog
from django.conf import settings
from django.db import transaction
from django.utils import timezone
from .models import TradingSignal, Order
from .circuit_breaker import CircuitBreaker

logger = structlog.get_logger(__name__)

class ExecutionManager:
    """
    مسؤول عن إرسال إشارات التداول إلى الوسيط بشكل آمن.
    """
    def __init__(self, signal: TradingSignal):
        self.signal = signal
        self.config = settings.TRADING_CONFIG
        self.mode = self.config.get("EXECUTION_MODE", "demo")

    def execute_trade(self):
        """
        ينفذ الصفقة: يتحقق من الضمانات، يرسل الأمر، ويسجل النتيجة.
        يستخدم قفلًا على مستوى قاعدة البيانات لضمان عدم التكرار (Idempotency).
        """
        try:
            with transaction.atomic():
                # قفل الصف لضمان عدم التكرار في بيئة متزامنة
                signal_to_execute = TradingSignal.objects.select_for_update().get(pk=self.signal.pk)

                # التحقق مما إذا كان قد تم التعامل مع هذه الإشارة بالفعل
                if signal_to_execute.status != "PENDING":
                    logger.warning("Signal is not in PENDING state. Skipping execution.", signal_id=signal_to_execute.id, status=signal_to_execute.status)
                    return
                
                # التحقق من قاطع الدائرة
                CircuitBreaker.check()

                order = self._process_order(signal_to_execute)
            
            # يتم إرسال المقاييس خارج المعاملة الذرية
            logger.info("Trade execution process completed.", order_id=order.id, status=order.status)
            # Observability: Increment counters for monitoring
            # In a real system, this would integrate with Prometheus/StatsD
            logger.info("metric:execution_attempts_total", signal_type=self.signal.signal_type)
            if order.status == "FILLED":
                logger.info("metric:executions_filled_total")

        except PermissionError as e: # تم تفعيل قاطع الدائرة
            self._handle_cancellation(str(e))
        except Exception as e:
            logger.exception("An unexpected error occurred during trade execution.", signal_id=self.signal.id)
            self._handle_failure(str(e))
            logger.error("metric:executions_failed_total")
    
    def _process_order(self, signal_to_execute: TradingSignal) -> Order:
        """يحتوي على منطق إنشاء الأمر وإرساله."""
        order = Order.objects.create(signal=signal_to_execute, status="SENT")

        if self.mode == "live":
            # قفل أمان صريح لمنع التشغيل الحقيقي عن طريق الخطأ
            logger.error("LIVE TRADING ATTEMPTED BUT IS DISABLED.", signal_id=signal_to_execute.id)
            raise RuntimeError("Live trading is disabled. Enable explicitly in settings after passing all acceptance tests.")
            # response = MT5Connector.send_order(...)
            # order.broker_order_id = response.get('id')
            # ... (logic to handle live response)
        else: # الوضع التجريبي (demo/dry-run)
            logger.info(
                "Executing trade in DEMO mode.",
                signal_id=signal_to_execute.id,
                asset=signal_to_execute.asset.symbol,
                type=signal_to_execute.signal_type,
                entry=signal_to_execute.entry_price
            )
            order.broker_order_id = f"demo_{signal_to_execute.id}"
            order.status = "FILLED"
            order.executed_at = timezone.now()
            order.broker_response = {"message": "Trade executed successfully in demo mode."}

        order.save()
        signal_to_execute.status = "EXECUTED"
        signal_to_execute.save()
        return order

    def _handle_cancellation(self, reason: str):
        """يعالج إلغاء الإشارة بسبب قاطع الدائرة."""
        logger.warning("Trade execution cancelled.", signal_id=self.signal.id, reason=reason)
        self.signal.status = "CANCELLED"
        self.signal.meta['cancel_reason'] = reason
        self.signal.save()

    def _handle_failure(self, error: str):
        """يعالج فشل الإشارة بسبب خطأ غير متوقع."""
        logger.error("Trade execution failed.", signal_id=self.signal.id, error=error)
        # لا نغير حالة الإشارة هنا، قد نرغب في إعادة محاولة التنفيذ
        Order.objects.filter(signal=self.signal).update(
            status="FAILED",
            broker_response={"error": error}
        )


        