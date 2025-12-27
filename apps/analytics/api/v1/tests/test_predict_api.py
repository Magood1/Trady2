# apps/api/v1/tests/test_predict_api.py
import pytest
from django.urls import reverse
from rest_framework.test import APIClient
from apps.mlops.services import clear_active_model_cache

@pytest.mark.django_db
class TestPredictAPI:
    def setup_method(self):
        clear_active_model_cache()

    def teardown_method(self):
        clear_active_model_cache()

    def test_predict_with_symbol_success(self, client: APIClient, asset, active_model):
        FeatureVector.objects.create(
            asset=asset, 
            timestamp=timezone.now(), 
            features={"atr": 0.1, "rsi": 55}
        )
        url = reverse("predict")
        response = client.post(url, {"symbol": asset.symbol}, format="json")
        
        assert response.status_code == 200
        data = response.json()
        assert "prob_up" in data
        assert data['model_version'] == active_model.version

    def test_predict_no_active_model(self, client: APIClient):
        ModelRegistry.objects.update(is_active=False)
        url = reverse("predict")
        response = client.post(url, {"symbol": "ANY"}, format="json")
        assert response.status_code == 503
        assert "No active model" in response.json()['error']

    def test_predict_with_missing_feature(self, client: APIClient, asset, active_model):
        FeatureVector.objects.create(
            asset=asset,
            timestamp=timezone.now(),
            features={"atr": 0.1} # Missing 'rsi'
        )
        url = reverse("predict")
        response = client.post(url, {"symbol": asset.symbol}, format="json")
        assert response.status_code == 200
        assert "prob_up" in response.json()