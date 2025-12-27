import os
from celery import Celery


os.environ.setdefault("DJANGO_SETTINGS_MODULE", "project_trady2.settings")
app = Celery("project_trady2")
app.config_from_object("django.conf:settings", namespace="CELERY")
app.autodiscover_tasks()

