import streamlit as st
import requests
import pandas as pd


API_BASE_URL = "http://127.0.0.1:8000/api/employee"  # adjust as needed

def fetch_employee_full_detail(emp_id, start_date=None, end_date=None):
    params = {"employee_id": emp_id}
    if start_date:
        params["start_date"] = start_date
    if end_date:
        params["end_date"] = end_date
    try:
        response = requests.get(f"{API_BASE_URL}/full_detail/", params=params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching employee details: {e}")
        return None

def render_employee_full_detail(emp_id):
    st.write(f"Fetching full details for Employee #{emp_id}")
    data = fetch_employee_full_detail(emp_id)
    if data:
        st.write("### Employee Information")
        st.json(data.get("employee"))
        
        st.write("### Unavailability")
        unavail_df = pd.DataFrame(data.get("unavailability"))
        st.table(unavail_df)
        
        st.write("### Shift Assignments")
        shifts_df = pd.DataFrame(data.get("shift_assignments"))
        st.table(shifts_df)

# Example usage in Streamlit
selected_employee = st.selectbox("Select Employee ID", [1, 2, 3])
if st.button("Show Full Details"):
    render_employee_full_detail(selected_employee)


def fetch_employees():
    try:
        response = requests.get(f"{API_BASE_URL}/")
        response.raise_for_status()
        return response.json()  # Expecting a list of employees
    except Exception as e:
        st.error(f"Error fetching employees: {e}")
        return []

def fetch_employee_detail(emp_id):
    try:
        response = requests.get(f"{API_BASE_URL}/employee/{emp_id}/")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching employee detail: {e}")
        return None

def fetch_employee_unavailability(emp_id):
    # If you have a custom endpoint for unavailability:
    # e.g., /api/employees/<id>/unavailability/
    try:
        response = requests.get(f"{API_BASE_URL}/employee/{emp_id}/unavailability/")
        response.raise_for_status()
        return response.json()  # Expecting a list of unavailability records
    except Exception as e:
        st.error(f"Error fetching unavailability: {e}")
        return []

def render_employee_list():
    st.title("Employee List")

    employees_data = fetch_employees()
    if isinstance(employees_data, dict) and "results" in employees_data:
        employees_data = employees_data["results"]  # If paginated
    df = pd.DataFrame(employees_data)

    if df.empty:
        st.write("No employees found.")
        return

    st.dataframe(df)

    # Let user select an employee to view detail
    # Alternatively, create a 'View' button for each row
    selected_id = st.selectbox(
        "Select an Employee ID to view details",
        df["id"].unique(),
        format_func=lambda x: f"Employee #{x}"
    )

    if st.button("View Employee Detail"):
        detail_data = fetch_employee_detail(selected_id)
        if detail_data:
            st.write("### Employee Detail")
            st.json(detail_data)  # or format as you like

            # Fetch and display unavailability
            unavail_list = fetch_employee_unavailability(selected_id)
            if unavail_list:
                st.write("### Unavailability")
                unavail_df = pd.DataFrame(unavail_list)
                st.table(unavail_df)

def render():
    render_employee_list()

# The Streamlit logic requires a main "run" point:
if __name__ == "__main__":
    render()
