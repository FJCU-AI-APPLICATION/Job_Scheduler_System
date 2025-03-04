from rest_framework.routers import DefaultRouter
from django.urls import path
from schedule.view import ScheduleDetailView, ScheduleListCreateView

router = DefaultRouter()

urlpatterns = [
    path('', ScheduleListCreateView.as_view(), name='schedule-list-create'),
    path('<int:pk>', ScheduleDetailView.as_view(), name='schedule-detail'),
]