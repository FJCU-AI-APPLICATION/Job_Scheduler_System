from datetime import date
from typing import List, Dict, Any

from datetime import datetime, timedelta

def generate_shifts_between(
    start_date: date, 
    end_date: date, 
    shift_list: List[Dict[str, str]]
) -> List[Dict[str, str]]:
    """
    Generates a list of scheduled shifts between start_date and end_date (inclusive)
    using the provided shift_list.

    Each shift in shift_list should be a dictionary with the following keys:
        - "shift_id": A unique identifier for the shift.
        - "start_time": A string representing the shift's start time (format "HH:MM:SS").
        - "end_time": A string representing the shift's end time (format "HH:MM:SS").

    For each day between start_date and end_date, this function combines the day with the shift
    times. If the shift's end time is less than or equal to its start time, the shift is assumed
    to cross midnight and its end datetime is adjusted to the next day.

    Returns:
        A list of dictionaries, each representing a scheduled shift with keys:
            - "shift_id": str,
            - "start_date": str ("YYYY-MM-DD HH:MM:SS"),
            - "end_date": str ("YYYY-MM-DD HH:MM:SS")
    """
    scheduled_shifts = []
    current_date = start_date

    while current_date <= end_date:
        for idx, shift in enumerate(shift_list, start=1):
            # If there's no shift_id in the shift dict, assign one using idx.
            shift_id = shift.get("shift_id", str(idx))
            shift_start = datetime.combine(
                current_date, 
                datetime.strptime(shift["start_time"], "%H:%M:%S").time()
            )
            shift_end = datetime.combine(
                current_date, 
                datetime.strptime(shift["end_time"], "%H:%M:%S").time()
            )
            if shift_end <= shift_start:
                shift_end += timedelta(days=1)
            
            scheduled_shifts.append({
                "shift_id": shift_id,
                "start_date": shift_start.strftime("%Y-%m-%d %H:%M:%S"),
                "end_date": shift_end.strftime("%Y-%m-%d %H:%M:%S")
            })
        current_date += timedelta(days=1)

    return scheduled_shifts


def list_to_dict_by_employee(data: List[Dict[str, Any]]) -> Dict[int, Dict[str, List[Dict[str, Any]]]]:
    """
    Transforms a list of unavailability records into a nested dictionary grouped by employee_id.
    
    Each key in the resulting dictionary is an employee_id.
    For each employee, the value is a dictionary with two keys: "DAY_OF_WEEK" and "DATE_RANGE".
    The value for each of these keys is a list of records that match that unavailability type.
    
    Args:
        data: A list of unavailability records, where each record is a dictionary with keys:
              "employee_id", "unavailability_type", "day_of_week", "start_date", "end_date", and "reason".
    
    Returns:
        A dictionary in which each key is an employee_id (int), and each value is a dictionary of the form:
            {
                "DAY_OF_WEEK": [list of DAY_OF_WEEK records],
                "DATE_RANGE": [list of DATE_RANGE records]
            }
    """
    result: Dict[int, Dict[str, List[Dict[str, Any]]]] = {}
    for record in data:
        emp_id = record["employee_id"]
        u_type = record["unavailability_type"]
        
        if emp_id not in result:
            # Initialize with empty lists for both types.
            result[emp_id] = {"DAY_OF_WEEK": [], "DATE_RANGE": []}
        
        # Append the record to the correct type list.
        if u_type in result[emp_id]:
            result[emp_id][u_type].append(record)
        else:
            # In case an unexpected type appears, you can choose to create a new key.
            result[emp_id][u_type] = [record]
    
    return result


# Example dictionary representing shifts grouped by policy.
policy_shift_dict = {
    4: [
        {'start_time': '09:00:00', 'end_time': '17:00:00'},
        {'start_time': '17:00:00', 'end_time': '23:00:00'},
        {'start_time': '23:00:00', 'end_time': '09:00:00'}
    ],
    7: [
        {'start_time': '07:00:00', 'end_time': '15:00:00'},
        {'start_time': '15:00:00', 'end_time': '23:00:00'},
        {'start_time': '23:00:00', 'end_time': '07:00:00'}
    ],
    8: [
        {'start_time': '09:00:00', 'end_time': '15:00:00'},
        {'start_time': '15:00:00', 'end_time': '23:00:00'},
        {'start_time': '23:00:00', 'end_time': '09:00:00'},
        {'start_time': '23:00:00', 'end_time': '09:00:00'}
    ]
}
unavailability_data = [
        {"employee_id": 1, "unavailability_type": "DAY_OF_WEEK", "day_of_week": 1, "start_date": None, "end_date": None, "reason": None},
        {"employee_id": 1, "unavailability_type": "DAY_OF_WEEK", "day_of_week": 2, "start_date": None, "end_date": None, "reason": None},
        {"employee_id": 1, "unavailability_type": "DATE_RANGE", "day_of_week": None, "start_date": "2024-04-01", "end_date": "2024-04-03", "reason": "Vacation"},
        {"employee_id": 2, "unavailability_type": "DAY_OF_WEEK", "day_of_week": 1, "start_date": None, "end_date": None, "reason": None},
        {"employee_id": 2, "unavailability_type": "DATE_RANGE", "day_of_week": None, "start_date": "2024-04-10", "end_date": "2024-04-15", "reason": "Sick"},
    ]

# Example usage:
if __name__ == "__main__":
    start = date(2024, 12, 1)
    end = date(2024, 12, 3)
    schedule:List[Dict] = generate_shifts_between(
        shift_list=policy_shift_dict[4],
        start_date=start, 
        end_date=end)
    
    grouped = list_to_dict_by_employee(unavailability_data)
    