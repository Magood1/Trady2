import uuid
from django.db import models


class ModelRegistry(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    version = models.CharField(max_length=50, unique=True, db_index=True)
    model_path = models.CharField(max_length=255, help_text="Path to the saved model file.")
    model_hash = models.CharField(max_length=64, help_text="SHA256 hash of the model file.")
    trained_at = models.DateTimeField(auto_now_add=True)
    is_active = models.BooleanField(default=False, help_text="Is this the active model for production?")
    training_params = models.JSONField(default=dict, help_text="Configuration used for training.")
    metrics = models.JSONField(default=dict, help_text="Validation/test metrics (e.g., accuracy, logloss).")
    feature_list = models.JSONField(default=list, help_text="List of feature names used for training.")

    class Meta:
        verbose_name = "Model Registry Entry"
        verbose_name_plural = "Model Registry"
        ordering = ["-trained_at"]

    def __str__(self) -> str:
        return f"Model v{self.version} ({'active' if self.is_active else 'inactive'})"
    

    