from rest_framework import generics, pagination
from schedule.serializer import ScheduleSerializer
from datetime import datetime, timedelta
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from policy.models import ShiftPolicy
from employee.model import EmployeeUnavailability
from datetime import datetime
from schedule.models import Schedule, ScheduleEmployee
from schedule.algorithms import process_schedule

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



class ScheduleComputeWithPolicyShiftsAPIView(APIView):
    """
    Compute scheduling assignments for a date range using a given policy.
    
    Expected JSON payload:
    {
        "policy_id": 1,
        "employee_ids": [101, 102, 103],
        "start_date": "2024-12-01",
        "end_date": "2024-12-07"
    }
    
    This view performs:
      - Query shift details (start_time, end_time, shift_index) for the policy.
      - For each day in the date range, check employee availability.
      - Organize the queried data and pass it to the scheduling algorithm.
    """
    
    def post(self, request, *args, **kwargs):
        data = request.data
        policy_id = data.get("policy_id")
        employee_ids = data.get("employee_ids", [])
        start_date_str = data.get("start_date")  # Format: "YYYY-MM-DD"
        end_date_str = data.get("end_date")      # Format: "YYYY-MM-DD"
        
        # Validate required fields.
        if not all([policy_id, employee_ids, start_date_str, end_date_str]):
            return Response(
                {"error": "Missing one or more required fields."},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Parse dates.
        try:
            start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
            end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()
        except Exception:
            return Response(
                {"error": "Invalid date format. Use YYYY-MM-DD."},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        if end_date < start_date:
            return Response(
                {"error": "end_date must be on or after start_date."},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Query shift details from ShiftPolicyDetail.
        shift_details_qs = ShiftPolicy.objects.filter(policy_id=policy_id).order_by("shift_index")
        shift_info = []
        for detail in shift_details_qs:
            shift_info.append({
                "shift_index": detail.shift_index,
                "start_time": detail.start_time.strftime("%H:%M:%S"),
                "end_time": detail.end_time.strftime("%H:%M:%S")
            })
        
        # For each day in the date range, query employee unavailability.
        daily_availability = []
        current_date = start_date
        while current_date <= end_date:
            day_of_week = current_date.isoweekday()  # 1=Monday, 7=Sunday
            available_employees = []
            unavailable_employees = []
            
            for emp_id in employee_ids:
                records = EmployeeUnavailability.objects.filter(employee_id=emp_id)
                is_unavailable = False
                for record in records:
                    if record.unavailability_type == "DAY_OF_WEEK" and record.day_of_week == day_of_week:
                        is_unavailable = True
                        break
                    elif record.unavailability_type == "DATE_RANGE" and record.start_date <= current_date <= record.end_date:
                        is_unavailable = True
                        break
                if is_unavailable:
                    unavailable_employees.append(emp_id)
                else:
                    available_employees.append(emp_id)
            
            daily_availability.append({
                "date": current_date.strftime("%Y-%m-%d"),
                "day_of_week": day_of_week,
                "available_employees": available_employees,
                "unavailable_employees": unavailable_employees
            })
            
            current_date += timedelta(days=1)
        
        # Package all queried data into a dictionary.
        queried_data = {
            "policy_id": policy_id,
            "start_date": start_date_str,
            "end_date": end_date_str,
            "shift_info": shift_info,
            "daily_availability": daily_availability
        }
        
        # Pass the queried data to the algorithm module.
        computed_schedule = process_schedule(queried_data)
        
        return Response(
            computed_schedule, 
            status=status.HTTP_200_OK
        )

class ConfirmScheduleAPIView(APIView):
    """
    This endpoint accepts a confirmed computed schedule and persists it in the database.
    
    Expected JSON payload:
    {
        "policy_id": 1,
        "start_date": "2024-12-01",
        "end_date": "2024-12-07",
        "schedule": [
             {
                 "date": "2024-12-01",
                 "shift_assignments": [
                      {
                          "shift_index": 1,
                          "start_time": "09:00:00",
                          "end_time": "13:00:00",
                          "assigned_employees": [101, 103]
                      },
                      {
                          "shift_index": 2,
                          "start_time": "13:00:00",
                          "end_time": "17:00:00",
                          "assigned_employees": [102]
                      }
                 ]
             },
             ...
        ]
    }
    
    For each shift assignment, this view creates a Schedule record and corresponding 
    ScheduleEmployee records in the database.
    """
    def post(self, request, *args, **kwargs):
        data = request.data
        policy_id = data.get("policy_id")
        start_date = data.get("start_date")
        end_date = data.get("end_date")
        schedule_list = data.get("schedule", [])
        
        if not all([policy_id, start_date, end_date, schedule_list]):
            return Response(
                {"error": "Missing required fields."},
                status=status.HTTP_400_BAD_REQUEST
                )
        
        created_schedule_ids = []
        
        for day in schedule_list:
            date_str = day.get("date")
            if not date_str:
                continue
            shift_assignments = day.get("shift_assignments", [])
            for assignment in shift_assignments:
                shift_index = assignment.get("shift_index")
                start_time = assignment.get("start_time")
                end_time = assignment.get("end_time")
                assigned_employees = assignment.get("assigned_employees", [])
                
                if not all([shift_index, start_time, end_time]):
                    continue
                
                schedule_record = Schedule.objects.create(
                    name=f"Policy {policy_id} Shift #{shift_index} on {date_str}",
                    description="Persisted from confirmed computed schedule.",
                    start_date=date_str,
                    start_time=start_time,
                    end_time=end_time,
                    manager=None  # Optionally assign a manager
                )
                created_schedule_ids.append(schedule_record.id)
                
                for emp_id in assigned_employees:
                    ScheduleEmployee.objects.create(
                        schedule=schedule_record,
                        employee_id=emp_id
                    )
        
        return Response(
            {
                "message": "Schedule confirmed and saved.", 
                "schedule_ids": created_schedule_ids
            },
            status=status.HTTP_201_CREATED
        )