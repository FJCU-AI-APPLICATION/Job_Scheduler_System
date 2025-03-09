import streamlit as st
import requests
import pandas as pd

API_BASE_URL = "http://127.0.0.1:8000/api"

def search_schedules(start_date, end_date):
    params = {}
    if start_date:
        params['start'] = start_date
    if end_date:
        params['end'] = end_date

    try:
        resp = requests.get(f"{API_BASE_URL}/schedule/", params=params)
        resp.raise_for_status()
        return resp.json()  # Could be list or paginated data
    except Exception as e:
        st.error(f"Error searching schedules: {e}")
        return []

def render_schedule_page():
    st.title("Schedule Search")

    # Date inputs
    start_date = st.date_input("Start date")
    end_date = st.date_input("End date")

    if st.button("Search"):
        schedules = search_schedules(start_date, end_date)
        print(schedules)
        if isinstance(schedules, dict) and "results" in schedules:
            schedules = schedules["results"]  # if paginated
        df = pd.DataFrame(schedules)
        if not df.empty:
            st.dataframe(df)
        else:
            st.info("No schedules found for this range.")

def render():
    render_schedule_page()

if __name__ == "__main__":
    render()
