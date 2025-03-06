# schedule/algorithm.py

def process_schedule(queried_data):
    """
    Process the queried data to compute a schedule.
    
    queried_data is a dict with:
      - 'shift_info': a list of shift detail dicts, each containing:
            { "shift_index": int, "start_time": "HH:MM:SS", "end_time": "HH:MM:SS" }
      - 'daily_availability': a list (one per day) of dicts with:
            {
              "date": "YYYY-MM-DD",
              "day_of_week": int,
              "available_employees": [list of emp_ids],
              "unavailable_employees": [list of emp_ids]
            }
    
    For demonstration, the algorithm assigns all available employees to each shift.
    """
    shift_info = queried_data.get("shift_info", [])
    daily_availability = queried_data.get("daily_availability", [])
    
    computed_schedule = {
        "policy_id": queried_data.get("policy_id"),
        "start_date": queried_data.get("start_date"),
        "end_date": queried_data.get("end_date"),
        "shift_details": shift_info,
        "schedule": []
    }
    
    # For each day, build shift assignments using available employee list.
    for day_data in daily_availability:
        day_schedule = {
            "date": day_data["date"],
            "day_of_week": day_data["day_of_week"],
            "shift_assignments": [],
            "unavailable_employees": day_data["unavailable_employees"]
        }
        for shift in shift_info:
            # Dummy assignment: assign all available employees to each shift.
            day_schedule["shift_assignments"].append({
                "shift_index": shift["shift_index"],
                "start_time": shift["start_time"],
                "end_time": shift["end_time"],
                "assigned_employees": day_data["available_employees"]
            })
        computed_schedule["schedule"].append(day_schedule)
    
    return computed_schedule
