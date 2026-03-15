import random

import numpy as np

from server.schemas import (
    ScheduleMetrics,
    SchedulingRequest,
    SchedulingResponse,
    ShiftAssignment,
)


def run_ga_inference(
    request: SchedulingRequest,
    generations: int = 100,
    pop_size: int = 50,
) -> SchedulingResponse:
    """Run Genetic Algorithm optimization for scheduling."""
    num_employees = len(request.employees)
    shifts_per_day = len(request.shifts)
    num_shifts = request.days * shifts_per_day
    shift_lengths = [s.length_hours for s in request.shifts]
    max_hours = {i: e.max_hours for i, e in enumerate(request.employees)}

    # Build unavailability lookup: (day, emp_idx) -> unavailable
    unavail_set: set[tuple[int, int]] = set()
    for u in request.unavailability:
        emp_idx = next(
            (i for i, e in enumerate(request.employees) if e.id == u.employee_id),
            None,
        )
        if emp_idx is not None:
            unavail_set.add((u.day, emp_idx))

    def fitness(schedule: np.ndarray) -> float:
        hours_assigned = np.zeros(num_employees)
        for i, emp_idx in enumerate(schedule):
            shift_type = i % shifts_per_day
            hours_assigned[emp_idx] += shift_lengths[shift_type]

        # Penalty for uneven distribution
        target = sum(shift_lengths) * request.days / num_employees
        balance_penalty = float(np.sum(np.abs(hours_assigned - target)))

        # Penalty for exceeding max hours
        exceed_penalty = sum(
            max(0, hours_assigned[i] - max_hours[i]) for i in range(num_employees)
        )

        # Penalty for back-to-back shifts by same employee
        back_to_back = sum(
            1 for i in range(num_shifts - 1) if schedule[i] == schedule[i + 1]
        )

        # Penalty for unavailable assignments
        unavail_penalty = 0
        for i, emp_idx in enumerate(schedule):
            day = i // shifts_per_day
            if (day, emp_idx) in unavail_set:
                unavail_penalty += 10

        return -(balance_penalty + exceed_penalty * 2 + back_to_back + unavail_penalty)

    # Generate initial population
    population = [
        np.random.randint(0, num_employees, size=num_shifts) for _ in range(pop_size)
    ]

    for _ in range(generations):
        population.sort(key=fitness, reverse=True)
        new_population = [population[0]]  # Keep best (elitism)

        for _ in range(pop_size - 1):
            parent1, parent2 = random.sample(population[: max(10, pop_size // 5)], 2)
            point = np.random.randint(1, num_shifts - 1)
            child = np.concatenate((parent1[:point], parent2[point:]))

            # Mutation
            if random.random() < 0.1:
                idx = np.random.randint(0, num_shifts)
                child[idx] = np.random.randint(0, num_employees)

            new_population.append(child)

        population = new_population

    # Best solution
    best = max(population, key=fitness)

    # Convert to response
    assignments: list[ShiftAssignment] = []
    hours_by_employee: dict[int, int] = {e.id: 0 for e in request.employees}

    for i, emp_idx in enumerate(best):
        day = i // shifts_per_day
        shift_idx = i % shifts_per_day
        emp = request.employees[int(emp_idx)]
        assignments.append(
            ShiftAssignment(day=day, shift_index=shift_idx, employee_id=emp.id)
        )
        hours_by_employee[emp.id] += shift_lengths[shift_idx]

    hours_values = list(hours_by_employee.values())
    fairness = 1.0 - (max(hours_values) - min(hours_values)) / max(max(hours_values), 1)

    return SchedulingResponse(
        schedule=assignments,
        metrics=ScheduleMetrics(
            fairness_score=round(fairness, 4),
            total_hours_by_employee=hours_by_employee,
        ),
    )
