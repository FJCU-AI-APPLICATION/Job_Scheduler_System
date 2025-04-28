from django.db import models

from employee.model import Employee
from rest_framework import serializers


class Schedule(models.Model):
    name = models.CharField(max_length=200)
    description = models.TextField()
    start_date = serializers.DateTimeField()
    end_date = serializers.DateTimeField()

    class Meta:
        db_table = "Schedule"

    def __str__(self):
        return f"{self.name} ({self.start_date} {self.start_time}-{self.end_time})"


class ScheduleEmployee(models.Model):
    """
    Bridge table for many-to-many: 
    Each schedule can have many employees assigned.
    """
    schedule = models.ForeignKey(Schedule, on_delete=models.CASCADE, related_name="schedule_assignments")
    employee = models.ForeignKey(Employee, on_delete=models.CASCADE, related_name="employee_assignments")
    assigned_date = serializers.DateTimeField()

    class Meta:
        db_table = "ScheduleEmployee"
        unique_together = ('schedule', 'employee')  # Prevent duplicates

    def __str__(self):
        return f"{self.employee.name} => {self.schedule.name}"
