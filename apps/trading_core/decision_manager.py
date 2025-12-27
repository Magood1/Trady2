# apps/trading_core/decision_manager.py
import structlog
from decimal import Decimal
from django.conf import settings
from django.utils import timezone
from apps.market_data.models import Asset
from apps.trading_core.models import TradingSignal
from apps.trading_core.execution_manager import ExecutionManager
from apps.analytics.services import OHLCVLoader
from apps.mlops.services import get_active_model
from apps.analytics.features.pipeline import FeaturePipeline

# Indicators
from ta.trend import EMAIndicator, ADXIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange

logger = structlog.get_logger(__name__)

class DecisionManager:
    def __init__(self, symbol: str, timeframe: str):
        self.symbol = symbol
        self.timeframe = timeframe # H1 usually
        self.asset = Asset.objects.get(symbol=symbol)
        
        # إعدادات المخاطر من الإعدادات العامة
        self.config = settings.TRADING_CONFIG
        self.account_balance = Decimal(str(self.config.get("ACCOUNT_BALANCE", 10000.0)))
        # نبدأ بمخاطرة محافظة جداً (0.25%) بسبب درس 2020
        self.risk_per_trade_pct = Decimal("0.0025") 

    def run_analysis(self):
        logger.info("Decision Pipeline Activated", symbol=self.symbol)

        # 1. Multi-Timeframe Data Loading (H1 & D1)
        loader = OHLCVLoader()
        end_utc = timezone.now()
        # نحتاج بيانات كافية لحساب EMA200 و ADX D1
        start_utc_h1 = end_utc - timezone.timedelta(days=20) 
        start_utc_d1 = end_utc - timezone.timedelta(days=100) 

        try:
            df_h1 = loader.load_dataframe(self.asset, self.timeframe, start_utc_h1, end_utc)
            # نحمل البيانات اليومية للفلتر الماكرو
            df_d1 = loader.load_dataframe(self.asset, 'D1', start_utc_d1, end_utc)
        except Exception as e:
            logger.error("Data Load Failed", error=str(e))
            return

        if len(df_h1) < 200 or len(df_d1) < 50:
            logger.warning("Insufficient Data Depth")
            return

        # 2. MACRO REGIME CHECK (The Shield against Whipsaws)
        # هذا الفلتر هو ما كان سينقذنا في 2020
        d1_adx = ADXIndicator(high=df_d1['high'], low=df_d1['low'], close=df_d1['close'], window=14).adx().iloc[-1]
        d1_atr = AverageTrueRange(high=df_d1['high'], low=df_d1['low'], close=df_d1['close'], window=14).average_true_range().iloc[-1]
        
        # الشرط: تقلب عالي (>20$) + اتجاه واضح (ADX > 20)
        # ملاحظة: خفضنا ADX قليلاً لضمان عدم تفويت بدايات الترند
        if d1_atr < 15.0 or d1_adx < 20.0:
            logger.info("REGIME FILTER: MARKET UNSAFE", d1_atr=d1_atr, d1_adx=d1_adx)
            return

        # 3. STRATEGY SIGNAL (H1 Trend Pullback)
        close = df_h1['close']
        
        h1_ema200 = EMAIndicator(close=close, window=200).ema_indicator().iloc[-1]
        h1_rsi = RSIIndicator(close=close, window=14).rsi().iloc[-1]
        h1_adx = ADXIndicator(high=df_h1['high'], low=df_h1['low'], close=close, window=14).adx().iloc[-1]
        
        last_close = close.iloc[-1]
        last_open = df_h1['open'].iloc[-1]
        
        # Logic: Uptrend + Oversold + Local Strength + Green Candle
        is_signal = (last_close > h1_ema200) and \
                    (h1_rsi < 45) and \
                    (h1_adx > 20) and \
                    (last_close > last_open)
        
        if not is_signal:
            return # Silent exit if no signal

        # 4. ML GATEKEEPER (The Quality Filter)
        model_info = get_active_model()
        if not model_info:
            logger.error("No Active ML Model! Aborting.")
            return
            
        model, registry = model_info
        
        try:
            # Prepare features exactly as trained
            features_df = FeaturePipeline.build_feature_dataframe(self.symbol, df_h1)
            latest_features = features_df.iloc[[-1]][registry.feature_list]
            
            ml_prob = model.predict_proba(latest_features)[0, 1]
            
            # عتبة القبول (من اختبار Q4 2024 الناجح)
            ML_THRESHOLD = 0.60
            
            logger.info("Signal Detected. ML Checking...", prob=ml_prob)
            
            if ml_prob < ML_THRESHOLD:
                logger.info("ML REJECTED SIGNAL", prob=ml_prob)
                return

        except Exception as e:
            logger.exception("ML Inference Error")
            return

        # 5. EXECUTION & RISK (Dynamic Sizing)
        # Volatility from Pipeline (Log Returns StdDev)
        current_vol = Decimal(str(features_df['vol_std'].iloc[-1]))
        current_price = Decimal(str(last_close))
        
        # Stop Distance logic: 1.5x Volatility
        # Ensure min stop for Gold is at least $2.5 to avoid noise stop-outs
        raw_stop_dist = current_price * current_vol * Decimal("1.5")
        stop_dist = max(raw_stop_dist, Decimal("2.5"))
        
        stop_loss = current_price - stop_dist
        
        # Take Profit: 2.0x Volatility (Reward Ratio ~1.33)
        # We aim for runs, so we give it space
        tp_dist = max(current_price * current_vol * Decimal("2.0"), Decimal("4.0"))
        take_profit = current_price + tp_dist
        
        # Position Sizing
        risk_amount = self.account_balance * self.risk_per_trade_pct
        if stop_dist == 0: return
        
        # Units = Risk / Distance
        position_size = risk_amount / stop_dist
        
        # Rounding (Gold typically 2 decimals, lots 2 decimals)
        position_size = position_size.quantize(Decimal("0.01"))

        # 6. FIRE!
        logger.warning(">>> EXECUTING LIVE TRADE <<<", symbol=self.symbol, size=position_size)
        
        signal, created = TradingSignal.objects.get_or_create(
            asset=self.asset,
            timestamp=timezone.now(),
            defaults={
                'signal_type': 'BUY',
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'position_size': position_size,
                'status': 'PENDING',
                'meta': {
                    'ml_prob': float(ml_prob),
                    'd1_adx': float(d1_adx),
                    'd1_atr': float(d1_atr),
                    'strategy': 'Hybrid_Trend_v1'
                }
            }
        )
        
        if created:
            ExecutionManager(signal).execute_trade()
            