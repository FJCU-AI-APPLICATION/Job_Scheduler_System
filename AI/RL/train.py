# train.py

import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import random
from model import DQN, ReplayBuffer
from environment import SchedulingEnv, NUM_EMPLOYEES
from typing import Tuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_dqn(
    episodes=2000,
    batch_size=64,
    gamma=0.99,
    lr=1e-3,
    epsilon_start=1.0,
    epsilon_end=0.01,
    epsilon_decay=500,
    checkpoint="model.pth"
):
    """
    Train a DQN agent on GPU (if available), save checkpoint.
    """
    env = SchedulingEnv()
    state_dim = env.num_employees + 1  # (shift_index + assigned_hours)
    action_dim = env.num_employees

    dqn = DQN(state_dim, action_dim).to(device)
    optimizer = optim.Adam(dqn.parameters(), lr=lr)
    replay_buffer = ReplayBuffer()

    epsilon = epsilon_start
    epsilon_decay_rate = (epsilon_start - epsilon_end) / epsilon_decay

    def epsilon_greedy_action(state_np: np.ndarray) -> int:
        if random.random() < epsilon:
            return random.randint(0, action_dim - 1)
        else:
            with torch.no_grad():
                state_v = torch.FloatTensor(state_np).unsqueeze(0).to(device)
                q_values = dqn(state_v)
                return int(torch.argmax(q_values, dim=1).item())

    def update_model():
        if len(replay_buffer) < batch_size:
            return
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        states_v = torch.FloatTensor(states).to(device)
        actions_v = torch.LongTensor(actions).to(device)
        rewards_v = torch.FloatTensor(rewards).to(device)
        next_states_v = torch.FloatTensor(next_states).to(device)
        dones_v = torch.FloatTensor(dones).to(device)

        # current Q
        q_values = dqn(states_v)
        q_values = q_values.gather(1, actions_v.unsqueeze(-1)).squeeze(-1)

        # next Q
        with torch.no_grad():
            next_q_values = dqn(next_states_v).max(1)[0]
            target_q = rewards_v + gamma * next_q_values * (1 - dones_v)

        loss = nn.MSELoss()(q_values, target_q)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    all_rewards = []
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            action = epsilon_greedy_action(state)
            next_state, reward, done, _ = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            # update
            update_model()

            # decay epsilon
            if epsilon > epsilon_end:
                epsilon -= epsilon_decay_rate
                if epsilon < epsilon_end:
                    epsilon = epsilon_end

        all_rewards.append(total_reward)
        if (episode+1) % 500 == 0:
            avg_r = np.mean(all_rewards[-500:])
            print(f"Episode {episode+1}, Epsilon={epsilon:.3f}, AvgReward={avg_r:.2f}")

    # Save checkpoint
    torch.save(dqn.state_dict(), checkpoint)
    print(f"\nTraining complete, model saved to {checkpoint}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10000, help="Number of training episodes.")
    parser.add_argument("--checkpoint", type=str, default="result/model.pth", help="Checkpoint file path.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epsilon_start", type=float, default=1.0)
    parser.add_argument("--epsilon_end", type=float, default=0.01)
    parser.add_argument("--epsilon_decay", type=float, default=500)

    args = parser.parse_args()

    train_dqn(
        episodes=args.episodes,
        batch_size=args.batch_size,
        gamma=args.gamma,
        lr=args.lr,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        checkpoint=args.checkpoint
    )

if __name__ == "__main__":
    main()
