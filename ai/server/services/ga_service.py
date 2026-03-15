import random

import numpy as np
from deap import algorithms, base, creator, tools

from server.schemas import (
    ScheduleMetrics,
    SchedulingRequest,
    SchedulingResponse,
    ShiftAssignment,
)

# Register DEAP types once at module level
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMax)


def run_ga_inference(
    request: SchedulingRequest,
    generations: int = 100,
    pop_size: int = 50,
) -> SchedulingResponse:
    """Run DEAP Genetic Algorithm optimization for scheduling."""
    num_employees = len(request.employees)
    shifts_per_day = len(request.shifts)
    num_shifts = request.days * shifts_per_day
    shift_lengths = [s.length_hours for s in request.shifts]
    max_hours = {i: e.max_hours for i, e in enumerate(request.employees)}

    # Build unavailability lookup
    unavail_set: set[tuple[int, int]] = set()
    for u in request.unavailability:
        emp_idx = next(
            (i for i, e in enumerate(request.employees) if e.id == u.employee_id),
            None,
        )
        if emp_idx is not None:
            unavail_set.add((u.day, emp_idx))

    def fitness(individual: list[int]) -> tuple[float]:
        hours_assigned = np.zeros(num_employees)
        for i, emp_idx in enumerate(individual):
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
            1 for i in range(num_shifts - 1) if individual[i] == individual[i + 1]
        )

        # Penalty for unavailable assignments
        unavail_penalty = 0
        for i, emp_idx in enumerate(individual):
            day = i // shifts_per_day
            if (day, emp_idx) in unavail_set:
                unavail_penalty += 10

        return (-(balance_penalty + exceed_penalty * 2 + back_to_back + unavail_penalty),)

    # Build DEAP toolbox for this specific request
    toolbox = base.Toolbox()
    toolbox.register("attr_emp", random.randint, 0, num_employees - 1)
    toolbox.register(
        "individual",
        tools.initRepeat,
        creator.Individual,
        toolbox.attr_emp,
        n=num_shifts,
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", fitness)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register(
        "mutate",
        tools.mutUniformInt,
        low=0,
        up=num_employees - 1,
        indpb=0.05,
    )
    toolbox.register("select", tools.selTournament, tournsize=3)

    hof = tools.HallOfFame(1)
    population = toolbox.population(n=pop_size)

    algorithms.eaSimple(
        population,
        toolbox,
        cxpb=0.7,
        mutpb=0.2,
        ngen=generations,
        halloffame=hof,
        verbose=False,
    )

    # Best solution from HallOfFame
    best = hof[0]

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
