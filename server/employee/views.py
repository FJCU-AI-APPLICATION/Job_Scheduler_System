from rest_framework import generics, pagination
from employee.model import Employee, EmployeeUnavailability
from employee.serializer import EmployeeSerializer, EmployeeUnavailabilitySerializer
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from schedule.models import ScheduleEmployee
from datetime import datetime

class EmployeePagination(pagination.PageNumberPagination):
    page_size = 5
    page_size_query_param = 'page_size'
    max_page_size = 20

# List and Create View
class EmployeeListCreateView(generics.ListCreateAPIView):
    queryset = Employee.objects.all()
    serializer_class = EmployeeSerializer
    pagination_class = EmployeePagination

# Retrieve, Update, Delete View
class EmployeeDetailView(generics.RetrieveUpdateDestroyAPIView):
    queryset = Employee.objects.all()
    serializer_class = EmployeeSerializer

class EmployeeUnavailabilityListCreateView(generics.ListCreateAPIView):
    """
    List all unavailable day records for a specific employee.
    The 'employee_id' query parameter is required.
    Also allows creating a new unavailable day record.
    """
    serializer_class = EmployeeUnavailabilitySerializer

    def get_queryset(self):
        employee_id = self.request.query_params.get('employee_id')
        if not employee_id:
            return Response(
                {"error": "The 'employee_id' query parameter is required."},
                status=status.HTTP_400_BAD_REQUEST
            )
        return EmployeeUnavailability.objects.filter(employee_id=employee_id)


class EmployeeUnavailabilityDetailView(generics.RetrieveUpdateDestroyAPIView):
    """
    Retrieve, update, or delete a specific EmployeeUnavailability record.
    """
    queryset = EmployeeUnavailability.objects.all()
    serializer_class = EmployeeUnavailabilitySerializer

class EmployeeShiftListView(APIView):
    """
    GET endpoint to retrieve the shifts assigned to a given employee.
    
    Query Parameters:
      - employee_id (required)
      - start_date (optional, in "YYYY-MM-DD" format)
      - end_date (optional, in "YYYY-MM-DD" format)
    
    The response returns a list of schedule assignments with the following details:
      - schedule name and description
      - schedule start_date, start_time, and end_time
      - assigned_date (when the assignment was created)
    """
    def get(self, request, *args, **kwargs):
        employee_id = request.query_params.get('employee_id')
        if not employee_id:
            return Response(
                {"error": "The 'employee_id' query parameter is required."},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Start with all schedule assignments for the employee.
        assignments = ScheduleEmployee.objects.filter(employee_id=employee_id).select_related('schedule')
        
        # Optionally filter by date range if provided.
        start_date = request.query_params.get('start_date')
        end_date = request.query_params.get('end_date')
        if start_date:
            assignments = assignments.filter(schedule__start_date__gte=start_date)
        if end_date:
            assignments = assignments.filter(schedule__start_date__lte=end_date)
        
        # Build the response context.
        result = []
        for assignment in assignments:
            sch = assignment.schedule
            result.append({
                "schedule_name": sch.name,
                "description": sch.description,
                "schedule_date": sch.start_date,  # Assuming schedule date is stored here.
                "start_time": sch.start_time.strftime("%H:%M:%S"),
                "end_time": sch.end_time.strftime("%H:%M:%S"),
                "assigned_date": assignment.assigned_date.strftime("%Y-%m-%d %H:%M:%S")
            })
        
        return Response(
            result, 
            status=status.HTTP_200_OK
        )

class EmployeeFullDetailView(APIView):
    """
    GET endpoint to retrieve full details for a specified employee.
    
    Query parameters:
      - employee_id (required)
      - start_date (optional, format "YYYY-MM-DD")
      - end_date (optional, format "YYYY-MM-DD")
    
    The response includes:
      - Basic employee information.
      - All unavailability records for the employee.
      - All shift assignments (from ScheduleEmployee) for the employee,
        optionally filtered by a date range.
    """
    def get(self, request, *args, **kwargs):
        employee_id = request.query_params.get("employee_id")
        if not employee_id:
            return Response({"error": "The 'employee_id' query parameter is required."},
                            status=status.HTTP_400_BAD_REQUEST)
        
        # Query basic employee info
        try:
            employee = Employee.objects.get(id=employee_id)
        except Employee.DoesNotExist:
            return Response({"error": "Employee not found."},
                            status=status.HTTP_404_NOT_FOUND)
        
        employee_data = EmployeeSerializer(employee).data
        
        # Query employee unavailability records
        unavailability_qs = EmployeeUnavailability.objects.filter(employee_id=employee_id)
        unavailability_data = EmployeeUnavailabilitySerializer(unavailability_qs, many=True).data
        
        # Query employee shift assignments from ScheduleEmployee
        assignments = ScheduleEmployee.objects.filter(employee_id=employee_id).select_related("schedule")
        
        # Optional: filter assignments by schedule date range if provided.
        start_date = request.query_params.get("start_date")
        end_date = request.query_params.get("end_date")
        if start_date:
            try:
                start_date_obj = datetime.strptime(start_date, "%Y-%m-%d").date()
                assignments = assignments.filter(assigned_date__gte=start_date_obj)
            except Exception as e:
                return Response({"error": "Invalid start_date format. Use YYYY-MM-DD."},
                                status=status.HTTP_400_BAD_REQUEST)
        if end_date:
            try:
                end_date_obj = datetime.strptime(end_date, "%Y-%m-%d").date()
                assignments = assignments.filter(assigned_date__lte=end_date_obj)
            except Exception as e:
                return Response({"error": "Invalid end_date format. Use YYYY-MM-DD."},
                                status=status.HTTP_400_BAD_REQUEST)
        
        assignment_list = []
        for assignment in assignments:
            sch = assignment.schedule
            assignment_list.append({
                "schedule_name": sch.name,
                "description": sch.description,
                "schedule_date": sch.start_date.strftime("%Y-%m-%d"),
                "start_time": sch.start_date.strftime("%H:%M:%S"),
                "end_time": sch.end_date.strftime("%H:%M:%S"),
                "assigned_date": assignment.assigned_date.strftime("%Y-%m-%d %H:%M:%S")
            })
        
        # Aggregate all data into one response.
        full_data = {
            "employee": employee_data,
            "unavailability": unavailability_data,
            "shift_assignments": assignment_list
        }
        
        return Response(full_data, status=status.HTTP_200_OK)


