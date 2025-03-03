from django.urls import path, include
from rest_framework.routers import DefaultRouter
from schedule.view import ScheduleViewSet

router = DefaultRouter()
router.register(r'schedule', ScheduleViewSet, basename='scheduler')

urlpatterns = [
    path('', include(router.urls)),
]