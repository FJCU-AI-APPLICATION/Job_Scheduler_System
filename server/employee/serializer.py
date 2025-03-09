from rest_framework import serializers
from employee.model import Employee, EmployeeUnavailability
from rest_framework import generics

class EmployeeSerializer(serializers.ModelSerializer):
    class Meta:
        model = Employee
        fields = '__all__'
        # or explicitly list fields:
        # fields = ['id', 'name', 'position', 'phone', 'date_joined']


class EmployeeUnavailabilitySerializer(serializers.ModelSerializer):
    class Meta:
        model = EmployeeUnavailability
        fields = '__all__'