"""Warm-start helpers for MaskablePPO: CP-SAT optimum enumeration + imitation BC.

Closes issue #17. Four public surfaces:

  enumerate_cpsat_optima(sp, config, n_solutions)
      Find up to n_solutions distinct schedules at the lex optimum.

  cpsat_schedules_to_transitions(env, schedules)
      Replay each schedule through env.step(); return imitation Transitions.

  bc_pretrain(policy, transitions, rng, n_batches, batch_size, lr)
      Train policy in-place via imitation.algorithms.bc.BC. Returns
      final loss + accuracy.

  WarmStartCallback (SB3 BaseCallback)
      Logs ΔHV + episode-end fitness components to TensorBoard each
      time env.step() emits a terminal-step info dict.
"""

from __future__ import annotations

import numpy as np
from imitation.algorithms.bc import BC
from imitation.data.types import Transitions
from ortools.sat.python import cp_model
from stable_baselines3.common.callbacks import BaseCallback

from ai.agents.environment import SchedulingEnv
from ai.domain.problem import SchedulingProblem
from ai.optimizers.cpsat import _build_model, _make_solver, _solve_stage
from ai.optimizers.result import CPSATConfig


# === CP-SAT optimum enumeration ===


class _SolutionCollector(cp_model.CpSolverSolutionCallback):
    """OR-Tools callback that harvests up to n_solutions distinct schedules."""

    def __init__(
        self,
        bundle,
        sp: SchedulingProblem,
        n_solutions: int,
    ):
        super().__init__()
        self._bundle = bundle
        self._sp = sp
        self._n_solutions = n_solutions
        self.schedules: list[list[int]] = []

    def on_solution_callback(self) -> None:
        T = self._sp.num_shifts
        E = self._sp.num_employees
        schedule: list[int] = []
        for t in range(T):
            assigned = next(
                (e for e in range(E) if self.Value(self._bundle.x[t][e]) == 1),
                0,
            )
            schedule.append(assigned)
        if schedule not in self.schedules:
            self.schedules.append(schedule)
        if len(self.schedules) >= self._n_solutions:
            self.StopSearch()


def enumerate_cpsat_optima(
    sp: SchedulingProblem,
    config: CPSATConfig,
    n_solutions: int = 20,
) -> list[list[int]]:
    """Find up to n_solutions distinct schedules at the lex optimum.

    Two-stage solve to find (b2b★, fairness_gap★); then rebuild model with
    both pinned as equality constraints and no objective; enumerate up to
    n_solutions distinct optimum-achieving schedules via
    enumerate_all_solutions=True and a solution callback.

    Stage 1 and stage 2 solves go through cpsat._solve_stage, so the same
    CPSATInfeasibleError / CPSATTimeoutError taxonomy is raised on failure.

    Reuses _build_model / _make_solver / _solve_stage from optimizers/cpsat.py.
    """
    # Stage 1: minimize b2b
    bundle_1 = _build_model(sp)
    bundle_1.model.Minimize(bundle_1.b2b_total)
    _, _, b2b_star, _ = _solve_stage(bundle_1, config, stage="b2b")

    # Stage 2: minimize fairness_gap under b2b ≤ b2b★
    bundle_2 = _build_model(sp)
    bundle_2.model.Add(bundle_2.b2b_total <= b2b_star)
    bundle_2.model.Minimize(bundle_2.fairness_gap)
    _, _, gap_star, _ = _solve_stage(bundle_2, config, stage="fairness")

    # Enumeration: rebuild model with both pinned, no objective
    bundle_e = _build_model(sp)
    bundle_e.model.Add(bundle_e.b2b_total == b2b_star)
    bundle_e.model.Add(bundle_e.fairness_gap == gap_star)
    collector = _SolutionCollector(bundle_e, sp, n_solutions)
    solver = _make_solver(config)
    solver.parameters.enumerate_all_solutions = True
    solver.parameters.num_search_workers = 1
    solver.Solve(bundle_e.model, collector)
    return collector.schedules


# === Trajectory generation ===


def cpsat_schedules_to_transitions(
    env: SchedulingEnv,
    schedules: list[list[int]],
) -> Transitions:
    """Replay each CP-SAT schedule through env.step(); capture
    (obs, action, next_obs, done) tuples in imitation's Transitions format.

    Asserts env.action_masks()[action] is True at every step — CP-SAT
    respects unavailability, so this should always hold.
    """
    obs_list, act_list, next_obs_list, done_list = [], [], [], []
    for schedule in schedules:
        obs, _ = env.reset(seed=0)
        for t, action in enumerate(schedule):
            assert env.action_masks()[action], (
                f"CP-SAT picked employee {action} at shift {t}, "
                f"but env.action_masks() says it's invalid."
            )
            obs_list.append(obs)
            act_list.append(action)
            next_obs, _r, terminated, _trunc, _info = env.step(action)
            next_obs_list.append(next_obs)
            done_list.append(terminated)
            obs = next_obs
    infos = np.empty(len(obs_list), dtype=object)
    infos.fill({})
    return Transitions(
        obs=np.array(obs_list, dtype=np.float32),
        acts=np.array(act_list, dtype=np.int64),
        infos=infos,
        next_obs=np.array(next_obs_list, dtype=np.float32),
        dones=np.array(done_list, dtype=bool),
    )


# === Behaviour cloning ===


def bc_pretrain(
    policy,
    transitions: Transitions,
    rng: np.random.Generator,
    n_batches: int = 5000,
    batch_size: int = 64,
    lr: float = 1e-3,
) -> dict[str, float]:
    """Train policy in-place via imitation.algorithms.bc.BC.

    Returns final loss + accuracy computed via a deterministic forward pass
    over the demonstrations (imitation's internal Logger.dump() clears
    name_to_value so the natively-recorded metrics aren't retrievable).
    """
    import torch

    bc_trainer = BC(
        observation_space=policy.observation_space,
        action_space=policy.action_space,
        demonstrations=transitions,
        policy=policy,
        batch_size=batch_size,
        optimizer_kwargs={"lr": lr},
        rng=rng,
    )
    bc_trainer.train(n_batches=n_batches)

    # Compute final metrics on the full demonstration set via public SB3 API.
    obs_tensor = torch.as_tensor(transitions.obs, dtype=torch.float32)
    acts_tensor = torch.as_tensor(transitions.acts, dtype=torch.int64)
    with torch.no_grad():
        _, log_probs, _ = policy.evaluate_actions(obs_tensor, acts_tensor)
        final_loss = float(-log_probs.mean())
    pred_actions, _ = policy.predict(transitions.obs, deterministic=True)
    final_accuracy = float((pred_actions == transitions.acts).mean())

    return {
        "final_loss": final_loss,
        "final_accuracy": final_accuracy,
    }


# === TensorBoard callback ===


class WarmStartCallback(BaseCallback):
    """Logs ΔHV + episode-end fitness components to TensorBoard.

    Reads the info dict emitted by SchedulingEnv.step() at terminal steps
    when pareto_reference is set.
    """

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "delta_hv" in info:
                self.logger.record("warm_start/delta_hv", info["delta_hv"])
                self.logger.record(
                    "warm_start/episode_violations", info["episode_violations"]
                )
                self.logger.record("warm_start/episode_b2b", info["episode_b2b"])
                self.logger.record(
                    "warm_start/episode_unfairness", info["episode_unfairness"]
                )
        return True
