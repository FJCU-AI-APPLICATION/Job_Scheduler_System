from django.urls import path
from schedule.view import ScheduleListCreateView, ScheduleDetailView, ScheduleComputeWithPolicyShiftsAPIView, ConfirmScheduleAPIView 

urlpatterns = [
    path('', ScheduleListCreateView.as_view(), name='schedule-list-create'),
    path('<int:pk>', ScheduleDetailView.as_view(), name='schedule-detail'),
    path('compute/', ScheduleComputeWithPolicyShiftsAPIView.as_view(), name='schedule-compute'),
    path('confirm/', ConfirmScheduleAPIView.as_view(), name='schedule-confirm'),
]
