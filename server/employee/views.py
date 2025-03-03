from rest_framework import generics, pagination
from employee.model import Employee
from employee.serializer import EmployeeSerializer

class EmployeePagination(pagination.PageNumberPagination):
    page_size = 5  # Number of employees per page
    page_size_query_param = 'page_size'  # Allow users to change page size via ?page_size=10
    max_page_size = 20  # Prevent large requests

# List and Create View
class EmployeeListCreateView(generics.ListCreateAPIView):
    queryset = Employee.objects.all()
    serializer_class = EmployeeSerializer
    pagination_class = EmployeePagination

# Retrieve, Update, Delete View
class EmployeeDetailView(generics.RetrieveUpdateDestroyAPIView):
    queryset = Employee.objects.all()
    serializer_class = EmployeeSerializer
