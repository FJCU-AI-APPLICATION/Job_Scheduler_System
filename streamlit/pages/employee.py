import streamlit as st
import requests
import pandas as pd

API_BASE_URL = "http://127.0.0.1:8000/api"  # Adjust to your Django server

def fetch_employees():
    try:
        response = requests.get(f"{API_BASE_URL}/employee/")
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
