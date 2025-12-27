# apps/market_data/tasks.py
import datetime
from typing import Dict, Optional

import structlog
from celery import shared_task
from django.utils import timezone

from apps.common.enums import Timeframe
from apps.market_data.services import ingest_ohlcv_data

logger = structlog.get_logger(__name__)


@shared_task(bind=True, default_retry_delay=300, max_retries=3)
def ingest_historical_data_task(
    self, symbol: str, timeframe_str: str, days_back: int = 0,
    start_utc: Optional[datetime.datetime] = None, end_utc: Optional[datetime.datetime] = None
) -> Dict:
    """
    يقوم بجلب البيانات التاريخية ويعيد قاموس سياق لسلسلة مهام Celery.
    هذه هي الخطوة الأولى التي تنشئ السياق الذي سيتم تمريره للمهام اللاحقة.
    """
    logger.info("Step 1/6: Starting data ingestion", symbol=symbol, timeframe=timeframe_str)
    
    if not end_utc:
        end_utc = timezone.now()
    if not start_utc:
        # إذا تم تحديد days_back، استخدمه؛ وإلا، افترض أن end_utc هو التاريخ الوحيد المهم
        if days_back > 0:
            start_utc = end_utc - datetime.timedelta(days=days_back)

    try:
        timeframe = Timeframe(timeframe_str)
        ingested_count = ingest_ohlcv_data(symbol, timeframe, start_utc, end_utc)
        
        # إنشاء وإعادة قاموس السياق لضمان تدفق البيانات في السلسلة
        return {
            "symbol": symbol,
            "timeframe": timeframe_str,
            "ingested_count": ingested_count,
            "start_utc": start_utc.isoformat() if start_utc else None,
            "end_utc": end_utc.isoformat() if end_utc else None
        }
    except Exception as exc:
        logger.error("Data ingestion task failed", exc_info=True, symbol=symbol)
        raise self.retry(exc=exc)


@shared_task(name="apps.market_data.tasks.trigger_decision_manager")
def trigger_decision_manager(prev_result: Dict) -> Dict:
    """
    يقوم بتشغيل DecisionManager. يستقبل السياق من المهمة السابقة في السلسلة.
    """
    symbol = prev_result.get("symbol")
    timeframe = prev_result.get("timeframe")
    
    if not symbol or not timeframe:
        logger.error("Could not trigger decision manager, missing context.", received=prev_result)
        # إعادة القاموس كما هو للسماح بالتصحيح دون كسر السلسلة
        return prev_result

    # تم تحديث الترقيم ليعكس موقعها في السلسلة الكاملة
    logger.info("Step 6/6: Triggering Decision Manager", symbol=symbol, timeframe=timeframe)
    from apps.trading_core.decision_manager import DecisionManager
    
    # DecisionManager مصمم لقراءة أحدث البيانات من قاعدة البيانات مباشرةً
    manager = DecisionManager(symbol, timeframe)
    manager.run_analysis()

    # --- الإصلاح المطبق ---
    # إعادة القاموس أمر حيوي للسماح باستمرار السلسلة إذا أضيفت مهام أخرى في المستقبل.
    # هذه هي أفضل ممارسة للمهام في سلسلة Celery.
    return prev_result