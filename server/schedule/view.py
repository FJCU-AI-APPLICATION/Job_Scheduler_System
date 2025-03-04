from rest_framework import generics, pagination
from schedule.model import Schedule
from schedule.serializer import ScheduleSerializer

class SchedulePagination(pagination.PageNumberPagination):
    page_size = 5
    page_size_query_param = 'page_size'
    max_page_size = 20


# List and Create View
class ScheduleListCreateView(generics.ListCreateAPIView):
    queryset = Schedule.objects.all()
    serializer_class = ScheduleSerializer
    pagination_class = SchedulePagination

# Retrieve, Update, Delete View
class  ScheduleDetailView(generics.RetrieveUpdateDestroyAPIView):
    queryset = Schedule.objects.all()
    serializer_class = ScheduleSerializer
