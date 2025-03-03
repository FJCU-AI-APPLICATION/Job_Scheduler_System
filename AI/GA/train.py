import numpy as np
import random

# Define scheduling parameters
NUM_EMPLOYEES = 7  # Number of part-time employees
NUM_SHIFTS = 30   # Number of shifts in a month
MAX_HOURS_PER_EMPLOYEE = 40

EMPLOYEE_MAP = ["member_A", "member_B", "member_C", "member_D", "member_E", "member_F", "member_G"]

# -----------------------------
# GA IMPLEMENTATION
# -----------------------------
def generate_initial_population(pop_size=50):
    return [np.random.randint(0, NUM_EMPLOYEES, size=NUM_SHIFTS) for _ in range(pop_size)]

def fitness(schedule, work_hours):
    # Count how many shifts each employee is assigned
    hours_assigned = np.bincount(schedule, minlength=NUM_EMPLOYEES)

    # 1) Penalty for deviating from the max allowed hours
    penalty = np.sum(np.abs(hours_assigned - MAX_HOURS_PER_EMPLOYEE))

    # 2) Penalty for consecutive (back-to-back) shifts by the same employee
    back_to_back_penalty = sum(schedule[i] == schedule[i + 1] for i in range(NUM_SHIFTS - 1))

    # 3) Optionally include difference from historical work_hours if you want
    #    to incorporate past distribution. For now, we'll skip it.

    # Overall: the smaller the penalty, the higher the fitness
    return - (penalty + back_to_back_penalty)

def crossover(parent1, parent2):
    point = np.random.randint(1, NUM_SHIFTS - 1)
    child = np.concatenate((parent1[:point], parent2[point:]))
    return child

def mutate(child, mutation_rate=0.1):
    if random.random() < mutation_rate:
        idx = np.random.randint(0, NUM_SHIFTS)
        child[idx] = np.random.randint(0, NUM_EMPLOYEES)
    return child

def genetic_algorithm(work_hours, generations=100, pop_size=50):
    population = generate_initial_population(pop_size)

    for _ in range(generations):
        # Sort the population by fitness in descending order
        population = sorted(population, key=lambda x: fitness(x, work_hours), reverse=True)
        new_population = [population[0]]  # Keep the best

        # Breed new individuals
        for _ in range(pop_size // 2):
            parent1, parent2 = random.sample(population[:10], 2)
            child = crossover(parent1, parent2)
            child = mutate(child)
            new_population.append(child)

        population = new_population

    # Return the best schedule found
    return population[0]

# Example: Historical work hours from previous data
work_hours = {
    'member_A': 182.0,
    'member_B': 126.0,
    'member_C': 114.0,
    'member_D': 101.0,
    'member_E': 43.0,
    'member_F': 202.0,
    'member_G': 72.0
}

# -----------------------------
# RUN GA & STORE RESULT
# -----------------------------
optimized_schedule = genetic_algorithm(work_hours)
print("Optimized Schedule (employee indices):", optimized_schedule)

# Example function to store the schedule
import csv

def store_schedule(schedule, filename="optimized_schedule.csv"):
    with open(filename, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Shift", "EmployeeIndex", "EmployeeName"])
        for shift_idx, emp_idx in enumerate(schedule):
            writer.writerow([shift_idx, emp_idx, EMPLOYEE_MAP[emp_idx]])
    print(f"Schedule saved to {filename}")

store_schedule(optimized_schedule)

# -----------------------------
# SIMPLE INFERENCE EXAMPLE
# -----------------------------
def analyze_schedule(schedule):
    # Count how many shifts each employee gets
    counts = np.bincount(schedule, minlength=NUM_EMPLOYEES)
    for i, c in enumerate(counts):
        print(f"{EMPLOYEE_MAP[i]} is assigned to {c} shifts.")
    return counts

print("\nInference about the final schedule:")
counts = analyze_schedule(optimized_schedule)
