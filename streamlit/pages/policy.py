import streamlit as st
import requests
import pandas as pd

API_BASE_URL = "http://127.0.0.1:8000/api"

def fetch_policies():
    try:
        resp = requests.get(f"{API_BASE_URL}/shiftpolicies/")
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"Error fetching policies: {e}")
        return []

def create_policy(name, description):
    data = {
        "policy_name": name,
        "description": description,
        "ai_model": ai_model_id  # Must be a valid AiModel ID
    }
    try:
        resp = requests.post(f"{API_BASE_URL}/shiftpolicies/", json=data)
        resp.raise_for_status()
        st.success("Policy created successfully!")
    except Exception as e:
        st.error(f"Error creating policy: {e}")

def update_policy(policy_id, name, description):
    data = {
        "policy_name": name,
        "description": description,
        # If you want to update AI model, add "ai_model" here
    }
    try:
        resp = requests.put(f"{API_BASE_URL}/shiftpolicies/{policy_id}/", json=data)
        resp.raise_for_status()
        st.success("Policy updated successfully!")
    except Exception as e:
        st.error(f"Error updating policy: {e}")

def fetch_policy_details(policy_id):
    # Could be embedded in the policy serializer or a separate endpoint
    # We'll assume shift_details is nested
    try:
        resp = requests.get(f"{API_BASE_URL}/shiftpolicies/{policy_id}/")
        resp.raise_for_status()
        return resp.json().get("shift_details", [])
    except Exception as e:
        st.error(f"Error fetching policy details: {e}")
        return []

def create_shift_detail(policy_id, index, start_time, end_time):
    data = {
        "policy": policy_id,
        "shift_index": index,
        "start_time": str(start_time),  # "HH:MM:SS"
        "end_time": str(end_time)
    }
    try:
        resp = requests.post(f"{API_BASE_URL}/shiftpolicydetails/", json=data)
        resp.raise_for_status()
        st.success("Shift detail created!")
    except Exception as e:
        st.error(f"Error creating shift detail: {e}")

def update_shift_detail(detail_id, index, start_time, end_time):
    data = {
        "shift_index": index,
        "start_time": str(start_time),
        "end_time": str(end_time)
    }
    try:
        resp = requests.put(f"{API_BASE_URL}/shiftpolicydetails/{detail_id}/", json=data)
        resp.raise_for_status()
        st.success("Shift detail updated!")
    except Exception as e:
        st.error(f"Error updating shift detail: {e}")

def render_policy_page():
    st.title("Shift Policy Management")

    # Section: Create a new policy
    st.subheader("Create New Policy")
    new_name = st.text_input("Policy Name")
    new_desc = st.text_area("Description")
    ai_model_id = st.number_input("AI Model ID", min_value=1, step=1)

    if st.button("Create Policy"):
        create_policy(new_name, new_desc, ai_model_id)

    # Section: Manage existing policies
    st.subheader("Existing Policies")
    policies_data = fetch_policies()
    if isinstance(policies_data, dict) and "results" in policies_data:
        policies_data = policies_data["results"]  # if paginated

    if policies_data:
        df_policies = pd.DataFrame(policies_data)
        st.dataframe(df_policies)

        selected_policy_id = st.selectbox(
            "Select a policy to edit",
            df_policies["id"].unique() if not df_policies.empty else []
        )

        selected_policy = df_policies[df_policies["id"] == selected_policy_id].iloc[0] if not df_policies.empty else None
        if selected_policy is not None:
            updated_name = st.text_input("Update Policy Name", value=selected_policy["policy_name"])
            updated_desc = st.text_area("Update Description", value=selected_policy["description"] or "")

            if st.button("Update Policy"):
                update_policy(selected_policy_id, updated_name, updated_desc)

            # Show shift details
            st.write("### Shift Details")
            policy_details = fetch_policy_details(selected_policy_id)
            df_details = pd.DataFrame(policy_details)
            st.table(df_details)

            # Add a new shift detail
            st.write("#### Add New Shift to Policy")
            new_shift_index = st.number_input("Shift Index", min_value=1, step=1)
            new_shift_start = st.time_input("Start Time")
            new_shift_end = st.time_input("End Time")
            if st.button("Add Shift Detail"):
                create_shift_detail(selected_policy_id, new_shift_index, new_shift_start, new_shift_end)

            # Optionally update an existing shift detail
            st.write("#### Update Existing Shift Detail")
            if not df_details.empty:
                detail_id = st.selectbox("Select Shift Detail ID to update", df_details["id"].unique())
                current_detail = df_details[df_details["id"] == detail_id].iloc[0]
                upd_index = st.number_input("Shift Index", min_value=1, value=current_detail["shift_index"], step=1)
                upd_start = st.time_input("Start Time", key="upd_start", value=pd.to_datetime(current_detail["start_time"]).time() if current_detail["start_time"] else None)
                upd_end = st.time_input("End Time", key="upd_end", value=pd.to_datetime(current_detail["end_time"]).time() if current_detail["end_time"] else None)

                if st.button("Update Shift Detail"):
                    update_shift_detail(detail_id, upd_index, upd_start, upd_end)

    else:
        st.info("No policies found. Please create one.")

def render():
    render_policy_page()

if __name__ == "__main__":
    render()
