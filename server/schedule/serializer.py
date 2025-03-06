from schedule.models import  Schedule
from rest_framework import serializers

class ScheduleSerializer(serializers.ModelSerializer):
    class Meta:
        model = Schedule
        fields = '__all__'
        # or explicitly list fields:
        # fields = ['id', 'name', 'position', 'phone', 'date_joined']