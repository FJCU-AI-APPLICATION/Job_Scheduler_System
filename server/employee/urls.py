from django.urls import path
from rest_framework.routers import DefaultRouter
from employee.views import EmployeeDetailView, EmployeeFullDetailView, EmployeeListCreateView, EmployeeShiftListView, EmployeeUnavailabilityDetailView, EmployeeUnavailabilityListCreateView

router = DefaultRouter()

urlpatterns = [
    path('', EmployeeListCreateView.as_view(), name='employee-list-create'),
    path('unavailabilities/', EmployeeUnavailabilityListCreateView.as_view(), name='employee-unavailability-list'),
    path('unavailabilities/<int:pk>/', EmployeeUnavailabilityDetailView.as_view(), name='employee-unavailability-detail'),
    path('employee_shifts/', EmployeeShiftListView.as_view(), name='employee-shifts'),
    path('full_detail/', EmployeeFullDetailView.as_view(), name='employee-full-detail'),

]