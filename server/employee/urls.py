from django.urls import path, include
from rest_framework.routers import DefaultRouter
from employee.views import EmployeeDetailView, EmployeeListCreateView

router = DefaultRouter()

urlpatterns = [
    path('', EmployeeListCreateView.as_view(), name='employee-list-create'),
    path('<int:pk>', EmployeeDetailView.as_view(), name='employee-detail'),
]