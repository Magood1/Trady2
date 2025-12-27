"""
URL configuration for project_trady2 project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.contrib import admin
from django.urls import include, path

from apps.market_data.views import HealthCheckAPIView

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/health/", HealthCheckAPIView.as_view(), name="health_check"),
    path("api/market-data/", include("apps.market_data.urls")),
    # أضف مسار واجهة برمجة تطبيقات التحليلات
    path("api/v1/analytics/", include("apps.analytics.api.v1.urls")),
]



