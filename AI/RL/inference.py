# visualize.py

import argparse
from model import DQN
import numpy as np
import matplotlib.pyplot as plt
from environment import NUM_EMPLOYEES, SchedulingEnv
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def inference(checkpoint="model.pth"):
    """
    Loads model checkpoint, runs inference, prints final schedule.
    """
    env = SchedulingEnv()
    state_dim = env.num_employees + 1
    action_dim = env.num_employees

    dqn = DQN(state_dim, action_dim).to(device)
    dqn.load_state_dict(torch.load(checkpoint, map_location=device))
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

    print("\n[Inference] Final schedule:", schedule)


def visualize(checkpoint="model.pth"):
    """
    Loads model checkpoint, runs inference, and visualizes distribution.
    """
    schedule = inference(checkpoint=checkpoint)
    counts = np.bincount(schedule, minlength=NUM_EMPLOYEES)

    plt.figure(figsize=(6,4))
    plt.bar(range(NUM_EMPLOYEES), counts, color='skyblue')
    plt.xticks(range(NUM_EMPLOYEES), [f"Emp{i}" for i in range(NUM_EMPLOYEES)])
    plt.ylabel("Number of Shifts")
    plt.title("Shift Distribution")
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="result/model.pth", help="Checkpoint file path.")
    args = parser.parse_args()

    visualize(checkpoint=args.checkpoint)

if __name__ == "__main__":
    main()
