import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, List, Deque
from collections import deque

# ==============
# SCHEDULING PARAMS
# ==============
NUM_EMPLOYEES = 7  # Default number of part-time employees
NUM_SHIFTS = 30    # Number of shifts in a month
MAX_HOURS = 40     # Max hours per employee
SHIFT_HOURS = 4    # Hours per shift

EMPLOYEE_MAP = ["member_A", "member_B", "member_C", "member_D", "member_E", "member_F", "member_G"]

# =====================
# ENVIRONMENT
# =====================

class SchedulingEnv:
    """
    A simple environment for scheduling shifts.
    State:
        (shift_index, hours_assigned[employee_0], ..., hours_assigned[employee_(N-1)])
    Action:
        employee_index (which employee takes the next shift)
    """
    def __init__(self, num_shifts=NUM_SHIFTS, num_employees=NUM_EMPLOYEES, max_hours=MAX_HOURS, shift_hours=SHIFT_HOURS):
        self.num_shifts = num_shifts
        self.num_employees = num_employees
        self.max_hours = max_hours
        self.shift_hours = shift_hours

        self.state = None
        self.previous_employee = None
        self.reset()

    def reset(self):
        # shift_index=0, all employees at 0 hours
        self.state = (0,) + tuple([0]*self.num_employees)
        self.previous_employee = None
        return self._get_observation()

    def step(self, action:int) -> Tuple[np.ndarray, float, bool, dict]:
        shift_index = self.state[0]
        hours = list(self.state[1:])

        reward = 0.0
        done = False

        # Penalty for consecutive assignment to the same employee
        if self.previous_employee is not None and action == self.previous_employee:
            reward -= 2.0

        # Assign SHIFT_HOURS to chosen employee
        hours[action] += self.shift_hours

        # Penalty if exceeding MAX_HOURS
        if hours[action] > self.max_hours:
            reward -= 5.0

        # small positive reward for a valid assignment
        reward += 1.0

        shift_index += 1
        self.previous_employee = action

        # if we've assigned all shifts, we are done
        if shift_index >= self.num_shifts:
            done = True

        self.state = (shift_index,) + tuple(hours)
        return self._get_observation(), reward, done, {}

    def _get_observation(self) -> np.ndarray:
        # Convert the environment's (shift_index, hours_0, ..., hours_n) into a float array
        return np.array(self.state, dtype=float)

    def set_partial_schedule(self, partial_schedule: List[int]):
        """
        Allows you to "inject" an existing partial schedule.
        partial_schedule: a list of assigned employees for the initial shifts.
        We'll fast-forward the environment state to that partial assignment.
        """
        shift_index = 0
        hours = [0]*self.num_employees
        self.previous_employee = None
        for emp in partial_schedule:
            # penalty if consecutive with previous
            if self.previous_employee is not None and emp == self.previous_employee:
                # we won't retroactively penalize in set, but keep track if needed.
                pass
            # assign shift hours
            hours[emp] += self.shift_hours
            shift_index += 1
            self.previous_employee = emp

        self.state = (shift_index,) + tuple(hours)
        return self._get_observation()

# =====================
# DQN NETWORK
# =====================

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.net(x)

# =====================
# REPLAY BUFFER
# =====================

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer: Deque = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states),
                np.array(actions),
                np.array(rewards, dtype=np.float32),
                np.array(next_states),
                np.array(dones, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)

# =====================
# DQN TRAINING
# =====================

def train_dqn(env: SchedulingEnv,
              episodes=2000,
              batch_size=64,
              gamma=0.99,
              lr=1e-3,
              epsilon_start=1.0,
              epsilon_end=0.01,
              epsilon_decay=500):
    """
    Trains a DQN agent on the scheduling environment.
    """
    state_dim = env.num_employees + 1  # shift_index + hours for each employee
    action_dim = env.num_employees

    dqn = DQN(state_dim, action_dim)
    optimizer = optim.Adam(dqn.parameters(), lr=lr)
    replay_buffer = ReplayBuffer()

    epsilon = epsilon_start
    epsilon_decay_rate = (epsilon_start - epsilon_end) / epsilon_decay

    def epsilon_greedy_action(state):
        if random.random() < epsilon:
            return random.randint(0, action_dim - 1)
        else:
            with torch.no_grad():
                state_v = torch.FloatTensor(state).unsqueeze(0)
                q_values = dqn(state_v)
                return int(torch.argmax(q_values, dim=1).item())

    def update_model():
        if len(replay_buffer) < batch_size:
            return
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        states_v = torch.FloatTensor(states)
        actions_v = torch.LongTensor(actions)
        rewards_v = torch.FloatTensor(rewards)
        next_states_v = torch.FloatTensor(next_states)
        dones_v = torch.FloatTensor(dones)

        # Current Q
        q_values = dqn(states_v)
        q_values = q_values.gather(1, actions_v.unsqueeze(-1)).squeeze(-1)

        # Next Q
        with torch.no_grad():
            next_q_values = dqn(next_states_v).max(1)[0]
            target_q = rewards_v + gamma * next_q_values * (1 - dones_v)

        loss = nn.MSELoss()(q_values, target_q)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # training loop
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

            # Update DQN
            update_model()

            # Decay epsilon
            if epsilon > epsilon_end:
                epsilon -= epsilon_decay_rate
                if epsilon < epsilon_end:
                    epsilon = epsilon_end

        all_rewards.append(total_reward)
        if (episode + 1) % 500 == 0:
            avg_reward = np.mean(all_rewards[-500:])
            print(f"Episode {episode+1}, Epsilon={epsilon:.3f}, Average Reward={avg_reward:.3f}")

    return dqn

# =====================
# BUILD SCHEDULE FROM TRAINED MODEL
# =====================
def build_schedule_from_dqn(env: SchedulingEnv, dqn: DQN):
    """
    Runs a greedy policy in the environment to produce a final schedule.
    """
    schedule = []
    state = env.reset()
    done = False
    while not done:
        with torch.no_grad():
            state_v = torch.FloatTensor(state).unsqueeze(0)
            q_values = dqn(state_v)
            action = int(torch.argmax(q_values, dim=1).item())
        schedule.append(action)
        next_state, _, done, _ = env.step(action)
        state = next_state
    return schedule

# ------------------------------
# INFERENCE MODES
# ------------------------------
def schedule_one_month(num_employees=7, num_shifts=30, episodes=2000):
    """
    1) Train & schedule N employees for 1 month.
    """
    env = SchedulingEnv(num_shifts=num_shifts, num_employees=num_employees)
    dqn_agent = train_dqn(env, episodes=episodes)
    schedule = build_schedule_from_dqn(env, dqn_agent)

    # Analyze results
    counts = np.bincount(schedule, minlength=num_employees)
    return schedule, counts

def add_new_employee_in_existing_schedule(base_schedule: List[int],
                                          base_num_employees=7,
                                          new_employee_index=7,
                                          total_shifts=30,
                                          episodes=2000):
    """
    2) Start with an existing schedule for (base_num_employees) employees,
       then add a new employee to the environment.

    We'll create a new environment with (base_num_employees+1) employees.
    We'll inject the base_schedule so the environment starts from the partial assigned state.
    Then we train or do inference for the remaining shifts.
    """
    # first, let's define the new environment (with 1 additional employee)
    new_num_employees = base_num_employees + 1
    env = SchedulingEnv(num_shifts=total_shifts, num_employees=new_num_employees)

    # inject partial schedule
    env.set_partial_schedule(base_schedule)

    # train on the new environment from that partial state forward
    dqn_agent = train_dqn(env, episodes=episodes)

    # now build the final schedule (which includes partial + newly assigned)
    # note that build_schedule_from_dqn re-starts env from scratch, so we need a custom approach.
    # We can modify it to preserve partial schedule or re-apply partial.

    # Let's do a custom method:
    final_schedule = base_schedule.copy()
    state = env.set_partial_schedule(base_schedule)
    done = False
    while not done:
        with torch.no_grad():
            state_v = torch.FloatTensor(state).unsqueeze(0)
            q_values = dqn_agent(state_v)
            action = int(torch.argmax(q_values, dim=1).item())
        final_schedule.append(action)
        next_state, _, done, _ = env.step(action)
        state = next_state

    counts = np.bincount(final_schedule, minlength=new_num_employees)
    return final_schedule, counts


if __name__ == "__main__":
    # ============ EXAMPLE 1: SCHEDULING ONE MONTH ============
    sch, cnts = schedule_one_month(num_employees=7, num_shifts=30, episodes=500)
    print("\n--- Inference #1: Scheduling 7 employees in 30 shifts ---")
    print("Schedule:", sch)
    for i, c in enumerate(cnts):
        print(f"Employee {i}: {c} shifts")

    # ============ EXAMPLE 2: ADD A NEW EMPLOYEE TO EXISTING SCHEDULE ============
    # Suppose we have an existing schedule from above for the first 20 shifts.
    base_schedule_20shifts = sch[:20]

    final_sched, final_cnts = add_new_employee_in_existing_schedule(
        base_schedule_20shifts,
        base_num_employees=7,
        new_employee_index=7,
        total_shifts=30,
        episodes=500
    )
    print("\n--- Inference #2: Add new employee to existing schedule ---")
    print("Final Schedule:", final_sched)
    for i, c in enumerate(final_cnts):
        print(f"Employee {i}: {c} shifts")
