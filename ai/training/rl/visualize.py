"""
visualize.py (Streamlit version)

Loads a trained SB3 model checkpoint, runs inference, and displays
the schedule in a table with a shift distribution bar chart.

Usage:
    streamlit run visualize.py
"""

import pandas as pd
import streamlit as st
import torch

from models.environment import EnvironmentConfig, SchedulingEnv
from models.problem import SchedulingProblem, jain_fairness_index
from server.services.model_registry import ModelRegistry


def run_inference(
    registry: ModelRegistry, checkpoint: str, config: EnvironmentConfig
) -> list[int]:
    """Run a trained SB3 model to produce a schedule."""
    env = SchedulingEnv(config)
    model = registry.load_model(checkpoint)

    schedule: list[int] = []
    obs, _ = env.reset()
    done = False

    while not done:
        action_masks = env.action_masks()
        try:
            action, _ = model.predict(obs, deterministic=True, action_masks=action_masks)
        except TypeError:
            action, _ = model.predict(obs, deterministic=True)
            if not action_masks[int(action)]:
                valid = torch.where(torch.as_tensor(action_masks))[0]
                action = valid[torch.randint(len(valid), (1,))].item() if len(valid) > 0 else action

        action = int(action)
        schedule.append(action)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    return schedule


def main() -> None:
    st.title("Scheduling Visualization (Streamlit)")
    st.write("Load a trained SB3 model checkpoint, run inference, and view the schedule.")

    config = EnvironmentConfig()
    problem = SchedulingProblem.from_config(config)

    checkpoint_dir = st.text_input("Checkpoint Directory", value="checkpoints")
    checkpoint_name = st.text_input("Checkpoint Name", value="best_model.zip")

    if st.button("Run Inference"):
        registry = ModelRegistry(checkpoint_dir)
        schedule = run_inference(registry, checkpoint_name, config)
        st.success("Inference complete!")

        # Build schedule table
        shift_names = ["Midnight(23-08)", "Day(08-16)", "Afternoon(16-23)"]
        data = []
        for i, emp_idx in enumerate(schedule):
            day_num = i // config.shifts_per_day + 1
            shift_type = i % config.shifts_per_day
            data.append(
                {
                    "Day": day_num,
                    "ShiftTime": (
                        shift_names[shift_type]
                        if shift_type < len(shift_names)
                        else f"Shift {shift_type}"
                    ),
                    "EmployeeIndex": emp_idx,
                    "EmployeeType": config.employee_types[emp_idx],
                }
            )
        df_schedule = pd.DataFrame(data)

        st.subheader("Monthly Schedule Table")
        st.dataframe(df_schedule)

        # Shift distribution
        counts = torch.bincount(
            torch.tensor(schedule), minlength=config.num_employees
        ).tolist()

        st.subheader("Shift Distribution")
        dist_df = pd.DataFrame(
            {
                "EmployeeIndex": range(config.num_employees),
                "EmployeeType": config.employee_types,
                "Count": counts,
            }
        )
        st.bar_chart(dist_df.set_index("EmployeeIndex").Count, use_container_width=True)
        st.write("Total shifts = ", len(schedule))

        # Quality metrics
        hours = problem.compute_hours(schedule)
        jain = jain_fairness_index(hours)
        b2b = problem.count_back_to_back(schedule)

        st.subheader("Schedule Quality")
        st.write(f"Jain's Fairness Index: {jain:.4f}")
        st.write(f"Back-to-back shifts: {b2b}")
        st.write("Hours per employee:")
        for i, h in enumerate(hours):
            emp_type = problem.employee_types[i]
            max_h = problem.max_hours[i]
            status = " **OVER**" if h > max_h else ""
            st.write(f"  Employee {i} ({emp_type}): {h:.0f}h / {max_h}h{status}")


if __name__ == "__main__":
    main()
