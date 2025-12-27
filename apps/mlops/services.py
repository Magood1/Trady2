# apps/mlops/services.py
import hashlib
import joblib
import os
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import lightgbm as lgb
from django.conf import settings
from django.core.exceptions import ObjectDoesNotExist

from .models import ModelRegistry

ModelObject = lgb.LGBMClassifier
ModelInfo = Tuple[ModelObject, ModelRegistry]

def save_model(
    model: ModelObject,
    config: Dict[str, Any],
    metrics: Dict[str, Any],
    feature_list: List[str],
    model_name_suffix: Optional[str] = None  # <-- إضافة جديدة
) -> ModelRegistry:
    """Saves a model artifact and registers it, with an optional version suffix."""
    mlops_cfg = config.get('mlops', {})
    output_dir_str = mlops_cfg.get('model_output_dir', 'mlops/models/')
    
    base_dir = Path(settings.BASE_DIR).resolve()
    output_dir = (base_dir / output_dir_str).resolve()

    if base_dir not in output_dir.parents and output_dir != base_dir:
        raise PermissionError(f"Model output directory '{output_dir}' must be within the project's BASE_DIR.")
        
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    # **تعديل مهم:** إضافة اللاحقة إلى الإصدار
    version_suffix = f"_{model_name_suffix}" if model_name_suffix else ""
    version = f"2.0.0+{timestamp}{version_suffix}" # <-- استخدام إصدار جديد 2.0.0
    
    model_file = output_dir / f"model_{version}.joblib"
    joblib.dump(model, model_file)

    with model_file.open('rb') as f:
        model_hash = hashlib.sha256(f.read()).hexdigest()

    relative_path = str(model_file.relative_to(base_dir))

    registry_entry = ModelRegistry.objects.create(
        version=version,
        model_path=relative_path,
        model_hash=model_hash,
        training_params=config,
        metrics=metrics,
        feature_list=feature_list,
        is_active=False
    )
    return registry_entry

@lru_cache(maxsize=4)
def get_active_model() -> Optional[ModelInfo]:
    """Loads the currently active model from the registry using a cache."""
    try:
        registry_entry = registry_entry = ModelRegistry.objects.filter(is_active=True).first()
        #ModelRegistry.objects.get(is_active=True)
        model_path = Path(settings.BASE_DIR) / registry_entry.model_path
        model = joblib.load(model_path)
        return model, registry_entry
    except (ObjectDoesNotExist, FileNotFoundError):
        return None

def get_model_by_version(version: str) -> Optional[ModelInfo]:
    """Loads a specific model version from the registry."""
    try:
        registry_entry = ModelRegistry.objects.get(version=version)
        model_path = Path(settings.BASE_DIR) / registry_entry.model_path
        model = joblib.load(model_path)
        return model, registry_entry
    except (ObjectDoesNotExist, FileNotFoundError):
        return None

def clear_active_model_cache():
    """Explicitly clears the LRU cache for get_active_model."""
    get_active_model.cache_clear()

    