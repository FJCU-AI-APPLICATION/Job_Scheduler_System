# myapp/business_logic.py
from datetime import timedelta
from typing import List, Dict, Any
import numpy as np
from schedule.algorithms import run_algorithm
from employee.model import EmployeeUnavailability
from policy.models import ShiftPolicy  # Import the algorithm dispatcher

# Type aliases for clarity.
ShiftInfo = Dict[str, str]         # e.g., {"shift_id": "1", "start_time": "09:00:00", "end_time": "17:00:00"}
DailyAvailability = Dict[str, Any]   # e.g., {"date": "2024-12-01", "day_of_week": 1, "available_employees": [101, 102]}
QueriedData = Dict[str, Any]
Matrix = np.ndarray  # NumPy array used as a tensor

def get_policy_shift_matrix(shift: List[Dict]) -> np.ndarray:
    """
    Retrieves shift details for a given policy and generates a NumPy matrix representing the shifts.
    
    Input:
      policy_id: int - The ID of the policy to retrieve shifts for.
    
    Output:
      A NumPy matrix (dtype=object) where each row represents a shift and contains:
         [shift_id (str), start_time (str), end_time (str)]
    """
    shift_qs = ShiftPolicy.objects.filter(policy_id=policy_id).order_by("start_time")
    num_shifts = shift_qs.count()
    
    for i, shift in enumerate(shift_qs):
        matrix[i, 0] = str(shift.id)
        matrix[i, 1] = shift.start_time.strftime("%H:%M:%S")
        matrix[i, 2] = shift.end_time.strftime("%H:%M:%S")
    
    return matrix

def compute_schedule_business_logic(policy_id: int, employee_ids: List[int], start_date, end_date, algorithm_type: str) -> Dict[str, Any]:
    """
    Executes the scheduling business logic:
      - Query shift details and employee availability.
      - Preprocess queried data into a NumPy matrix.
      - Run a scheduling algorithm based on the chosen type.
      - Postprocess the computed matrix into a JSON-friendly format.
    """
    # 1. Retrieve the shift matrix using the policy.
    shift_matrix: Matrix = get_policy_shift_matrix(policy_id)
    
    # Optionally, convert the shift matrix into a list of dictionaries if needed for further processing.
    # For example:
    shift_info: List[ShiftInfo] = []
    for row in shift_matrix:
        shift_info.append({
            "shift_id": row[0],
            "start_time": row[1],
            "end_time": row[2]
        })
    
    # 2. Build daily availability.
    daily_availability: List[DailyAvailability] = []
    current_date = start_date
    while current_date <= end_date:
        day_of_week = current_date.isoweekday()  # 1=Monday, 7=Sunday
        available_employees = []
        
        for emp_id in employee_ids:
            records = EmployeeUnavailability.objects.filter(employee_id=emp_id)
            is_unavailable = False
            for record in records:
                if record.unavailability_type == "DAY_OF_WEEK" and record.day_of_week == day_of_week:
                    is_unavailable = True
                    break
                elif (record.unavailability_type == "DATE_RANGE" and 
                      record.start_date <= current_date <= record.end_date):
                    is_unavailable = True
                    break
            if not is_unavailable:
                available_employees.append(emp_id)
                
        daily_availability.append({
            "date": current_date.strftime("%Y-%m-%d"),
            "day_of_week": day_of_week,
            "available_employees": available_employees
        })
        current_date += timedelta(days=1)
    
    # Package queried data.
    queried_data: QueriedData = {
        "policy_id": policy_id,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "shift_info": shift_info,
        "daily_availability": daily_availability
    }
    
    # 3. Preprocess: transform queried data into a NumPy matrix.
    matrix: Matrix = preprocess_data(queried_data)
    
    # 4. Run the selected scheduling algorithm.
    computed_matrix: Matrix = run_algorithm(matrix, shift_info, algorithm_type)
    
    # 5. Postprocess: convert the computed matrix to JSON format.
    json_output = postprocess_data(computed_matrix)
    return json_output

def preprocess_data(query_data: QueriedData) -> Matrix:
    """
    Transforms queried data into a NumPy matrix.
    
    Each row represents a day and contains:
      [date (str), day_of_week (int), available_employees (List[int])]
    """
    daily_availability = query_data.get("daily_availability", [])
    num_days = len(daily_availability)
    matrix = np.empty((num_days, 3), dtype=object)
    
    for i, day in enumerate(daily_availability):
        matrix[i, 0] = day["date"]
        matrix[i, 1] = day["day_of_week"]
        matrix[i, 2] = day["available_employees"]
        
    return matrix

def postprocess_data(matrix: Matrix) -> Dict[str, Any]:
    """
    Converts the computed schedule matrix into a JSON-friendly dictionary.
    
    Each row of the matrix is expected to have:
      [shift_id, start_time, end_time, date, assigned_employees (List[int])]
    """
    schedule = []
    for row in matrix:
        shift_id, start_time, end_time, date, assigned_employees = row
        schedule.append({
            "shift_id": shift_id,
            "start_time": start_time,
            "end_time": end_time,
            "date": date,
            "users": assigned_employees
        })
    return {"schedule": schedule}
