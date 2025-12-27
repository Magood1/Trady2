# apps/api/v1/endpoints/predict.py
import time
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import pandas as pd
import numpy as np
import structlog
from django.conf import settings

from apps.mlops.services import get_active_model
from apps.trading_core.models import FeatureVector
from apps.market_data.models import Asset

logger = structlog.get_logger(__name__)


def _prepare_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    يضمن أن جميع الأعمدة في DataFrame رقمية ويتعامل مع القيم اللانهائية (inf) والقيم غير الرقمية (NaN).
    """
    return df.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)


class PredictAPIView(APIView):
    """
    نقطة نهاية (Endpoint) للـ API تقوم بتلقي بيانات الأصول أو متجهات الميزات
    وتُرجع توقعًا لاحتمالية ارتفاع السعر باستخدام نموذج تعلم الآلة النشط.
    """
    def post(self, request, *args, **kwargs):
        start_time = time.perf_counter()
        
        model_info = get_active_model()
        if not model_info:
            logger.warning("Prediction API called but no active model found.")
            return Response({"error": "No active model is available for prediction."}, status=status.HTTP_503_SERVICE_UNAVAILABLE)
        
        model, registry = model_info

        try:
            if 'symbol' in request.data:
                asset = Asset.objects.get(symbol__iexact=request.data['symbol'])
                fv = FeatureVector.objects.filter(asset=asset).latest('timestamp')
                features_dict = fv.features
            elif 'feature_vector' in request.data:
                features_dict = request.data['feature_vector']
            else:
                return Response({"error": "Payload must contain 'symbol' or 'feature_vector'."}, status=status.HTTP_400_BAD_REQUEST)

            features = pd.DataFrame([features_dict])
            expected_cols = registry.feature_list
            
            # التأكد من وجود جميع الأعمدة المتوقعة في DataFrame
            for col in expected_cols:
                if col not in features.columns:
                    features[col] = 0.0
            
            # ترتيب الأعمدة وضمان أن تكون البيانات رقمية ونظيفة
            features = features[expected_cols]
            features = _prepare_numeric_df(features)

            prob_up = float(model.predict_proba(features)[0, 1])
        
        except Asset.DoesNotExist:
            return Response({"error": f"Asset '{request.data['symbol']}' not found."}, status=status.HTTP_404_NOT_FOUND)
        except FeatureVector.DoesNotExist:
            return Response({"error": f"No feature vectors found for asset '{request.data['symbol']}'."}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            logger.exception("Prediction failed due to an internal error.")
            return Response({"error": f"Model prediction failed: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        end_time = time.perf_counter()
        
        return Response({
            "prob_up": prob_up,
            "model_version": registry.version,
            "runtime_ms": (end_time - start_time) * 1000,
            "demo_mode": bool(getattr(settings, "DEBUG", True)),
        })