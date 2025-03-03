"""
visualize.py (Streamlit version)

This script replaces the previous matplotlib-based visualization.
We use Streamlit to:
  1) Load a checkpoint
  2) Run inference
  3) Display the entire month schedule in a table (Day, ShiftTime, EmployeeIndex, EmployeeType)
  4) Show a bar chart of total shifts per employee

Usage:
    streamlit run visualize.py

"""

from AI.RL.environment import EMPLOYEE_TYPES, NUM_EMPLOYEES, SchedulingEnv
from model import DQN
import streamlit as st
import torch
import numpy as np
import pandas as pd

from typing import List


########################################
# INFERENCE LOGIC
########################################

def run_inference(checkpoint_path: str) -> List[int]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = SchedulingEnv()
    state_dim = env.num_employees + 1
    action_dim = env.num_employees

    dqn = DQN(state_dim, action_dim).to(device)
    dqn.load_state_dict(torch.load(checkpoint_path, map_location=device))
    dqn.eval()

    schedule = []
    state = env.reset()
    done = False
    while not done:
        with torch.no_grad():
            state_v = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = dqn(state_v)
            action = int(torch.argmax(q_values, dim=1).item())
        schedule.append(action)
        next_state, _, done, _ = env.step(action)
        state = next_state

    return schedule

########################################
# STREAMLIT APP
########################################

def main():
    st.title("Scheduling Visualization (Streamlit)")
    st.write("This app loads a trained DQN model checkpoint, runs inference, and displays the final schedule.")

    # UI elements
    checkpoint_path = st.text_input("Checkpoint Path", value="model.pth")
    if st.button("Run Inference"):
        schedule = run_inference(checkpoint_path)
        st.success("Inference complete!")
        # Build a DataFrame showing Day, Shift, Assigned Employee, Employee Type
        data = []
        shift_names = ["Midnight(23-08)", "Day(08-16)", "Afternoon(16-23)"]
        for i, emp_idx in enumerate(schedule):
            day_num = i // 3 + 1
            shift_type = i % 3
            data.append({
                "Day": day_num,
                "ShiftTime": shift_names[shift_type],
                "EmployeeIndex": emp_idx,
                "EmployeeType": EMPLOYEE_TYPES[emp_idx]
            })
        df_schedule = pd.DataFrame(data)

        st.subheader("Monthly Schedule Table")
        st.dataframe(df_schedule)

        # Summarize counts
        counts = np.bincount(schedule, minlength=NUM_EMPLOYEES)

        st.subheader("Shift Distribution")
        dist_df = pd.DataFrame({
            "EmployeeIndex": range(NUM_EMPLOYEES),
            "EmployeeType": [EMPLOYEE_TYPES[i] for i in range(NUM_EMPLOYEES)],
            "Count": counts
        })
        st.bar_chart(dist_df.set_index("EmployeeIndex")["Count"], use_container_width=True)

        st.write("Total shifts = ", len(schedule))

if __name__ == "__main__":
    main()
