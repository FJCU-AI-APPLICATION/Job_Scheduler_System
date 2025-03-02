from rest_framework import generics
from employee.model import Employee
from employee.serializer import EmployeeSerializer

# List and Create View
class EmployeeListCreateView(generics.ListCreateAPIView):
    queryset = Employee.objects.all()
    serializer_class = EmployeeSerializer

# Retrieve, Update, Delete View
class EmployeeDetailView(generics.RetrieveUpdateDestroyAPIView):
    queryset = Employee.objects.all()
    serializer_class = EmployeeSerializer
