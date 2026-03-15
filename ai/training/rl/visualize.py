"""
visualize.py (Streamlit version)

Loads a trained SB3 model checkpoint, runs inference, and displays
the schedule in a table with a shift distribution bar chart.

Usage:
    streamlit run visualize.py
"""

import numpy as np
import pandas as pd
import streamlit as st
from sb3_contrib import MaskablePPO

from models.environment import EnvironmentConfig, SchedulingEnv


def run_inference(checkpoint_path: str, config: EnvironmentConfig) -> list[int]:
    env = SchedulingEnv(config)
    model = MaskablePPO.load(checkpoint_path)

    schedule = []
    obs, _ = env.reset()
    done = False

    while not done:
        action_masks = env.action_masks()
        try:
            action, _ = model.predict(obs, deterministic=True, action_masks=action_masks)
        except TypeError:
            action, _ = model.predict(obs, deterministic=True)

        action = int(action)
        schedule.append(action)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    return schedule


def main() -> None:
    st.title("Scheduling Visualization (Streamlit)")
    st.write("Load a trained SB3 model checkpoint, run inference, and view the schedule.")

    config = EnvironmentConfig()

    checkpoint_path = st.text_input("Checkpoint Path", value="checkpoints/best_model")
    if st.button("Run Inference"):
        schedule = run_inference(checkpoint_path, config)
        st.success("Inference complete!")

        data = []
        shift_names = ["Midnight(23-08)", "Day(08-16)", "Afternoon(16-23)"]
        for i, emp_idx in enumerate(schedule):
            day_num = i // config.shifts_per_day + 1
            shift_type = i % config.shifts_per_day
            data.append(
                {
                    "Day": day_num,
                    "ShiftTime": shift_names[shift_type] if shift_type < len(shift_names) else f"Shift {shift_type}",
                    "EmployeeIndex": emp_idx,
                    "EmployeeType": config.employee_types[emp_idx],
                }
            )
        df_schedule = pd.DataFrame(data)

        st.subheader("Monthly Schedule Table")
        st.dataframe(df_schedule)

        counts = np.bincount(schedule, minlength=config.num_employees)

        st.subheader("Shift Distribution")
        dist_df = pd.DataFrame(
            {
                "EmployeeIndex": range(config.num_employees),
                "EmployeeType": config.employee_types,
                "Count": counts,
            }
        )
        st.bar_chart(dist_df.set_index("EmployeeIndex")["Count"], use_container_width=True)
        st.write("Total shifts = ", len(schedule))


if __name__ == "__main__":
    main()
