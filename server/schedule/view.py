# app/views.py
from rest_framework import viewsets
from schedule.model import Schedule
from schedule.serializer import ScheduleSerializer

class ScheduleViewSet(viewsets.ModelViewSet):
    queryset = Schedule.objects.all()
    serializer_class = ScheduleSerializer