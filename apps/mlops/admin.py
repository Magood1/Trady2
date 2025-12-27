# apps/mlops/admin.py
from django.contrib import admin, messages
from django.db import transaction
from .models import ModelRegistry
from .services import clear_active_model_cache

@admin.register(ModelRegistry)
class ModelRegistryAdmin(admin.ModelAdmin):
    list_display = ('version', 'is_active', 'trained_at', 'model_hash')
    list_filter = ('is_active',)
    search_fields = ('version',)
    readonly_fields = ('id', 'trained_at', 'model_path', 'model_hash', 'training_params', 'metrics', 'feature_list')
    actions = ['activate_model']

    def activate_model(self, request, queryset):
        if queryset.count() != 1:
            self.message_user(request, "Please select exactly one model to activate.", level=messages.WARNING)
            return
        
        with transaction.atomic():
            ModelRegistry.objects.filter(is_active=True).update(is_active=False)
            model = queryset.first()
            model.is_active = True
            model.save()
        
        clear_active_model_cache()
        
        self.message_user(request, f"Model v{model.version} has been activated.", level=messages.SUCCESS)

    activate_model.short_description = "Activate selected model for production"