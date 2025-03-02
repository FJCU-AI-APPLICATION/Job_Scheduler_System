from rest_framework import serializers
from employee.model import Employee

class EmployeeSerializer(serializers.ModelSerializer):
    class Meta:
        model = Employee
        fields = '__all__'
        # or explicitly list fields:
        # fields = ['id', 'name', 'position', 'phone', 'date_joined']