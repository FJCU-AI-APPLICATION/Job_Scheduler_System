"""Train an RL agent for shift scheduling using Stable-Baselines3.

Supports MaskablePPO (default), DQN, PPO, and A2C. MaskablePPO is recommended
as it handles unavailability constraints natively during training via action masking.

Usage:
    python -m training.rl --algorithm maskable_ppo --total-timesteps 500000
    python -m training.rl --algorithm dqn --total-timesteps 200000 --lr 1e-4
"""

import argparse
import json
import sys
from pathlib import Path

from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from ai.agents.environment import EnvironmentConfig, SchedulingEnv
from ai.agents.registry import ALGORITHM_MAP


def make_env(config: EnvironmentConfig) -> SchedulingEnv:
    return SchedulingEnv(config)


def train(
    algorithm: str = "maskable_ppo",
    total_timesteps: int = 500_000,
    learning_rate: float = 3e-4,
    checkpoint_dir: str = "checkpoints",
    tb_log_dir: str = "logs",
    eval_freq: int = 5_000,
    checkpoint_freq: int = 10_000,
    net_arch: list[int] | None = None,
    fairness_alpha: float = 2.0,
    warm_start: str | None = None,
    warm_start_solutions: int = 20,
    bc_steps: int = 5000,
    bc_lr: float = 1e-3,
    pareto_ref: str | None = None,
) -> None:
    if algorithm not in ALGORITHM_MAP:
        print(f"Unknown algorithm: {algorithm}. Choose from: {list(ALGORITHM_MAP.keys())}")
        sys.exit(1)

    # Warm-start scope guard.
    if warm_start == "cpsat" and algorithm != "maskable_ppo":
        raise ValueError(
            f"--warm-start cpsat only supported for maskable_ppo, got {algorithm}. "
            "Drop --warm-start or switch to --algorithm maskable_ppo."
        )

    if net_arch is None:
        net_arch = [128, 64]

    # Load Pareto reference if provided.
    pareto_reference = None
    if pareto_ref is not None:
        payload = json.loads(Path(pareto_ref).read_text())
        pareto_reference = [tuple(p) for p in payload["points"]]
        if not pareto_reference:
            raise ValueError(
                f"--pareto-ref {pareto_ref} has empty 'points'. "
                "Retrain CCMO with larger budget."
            )

    config = EnvironmentConfig(
        fairness_alpha=fairness_alpha,
        pareto_reference=pareto_reference,
    )
    env = make_env(config)
    eval_env = make_env(config)

    algo_cls = ALGORITHM_MAP[algorithm]
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    model = algo_cls(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=learning_rate,
        tensorboard_log=tb_log_dir,
        policy_kwargs=dict(net_arch=net_arch),
    )

    # Warm-start: BC on CP-SAT trajectories.
    if warm_start == "cpsat":
        import numpy as np

        from ai.agents.warm_start import (
            bc_pretrain,
            cpsat_schedules_to_transitions,
            enumerate_cpsat_optima,
        )
        from ai.domain.problem import SchedulingProblem
        from ai.optimizers.result import CPSATConfig

        problem = SchedulingProblem.from_config(config)
        cpsat_config = CPSATConfig()
        schedules = enumerate_cpsat_optima(
            problem, cpsat_config, n_solutions=warm_start_solutions
        )
        print(f"Enumerated {len(schedules)} CP-SAT optimum schedules for BC pretraining.")
        transitions = cpsat_schedules_to_transitions(env, schedules)
        rng = np.random.default_rng(seed=0)
        metrics = bc_pretrain(
            model.policy, transitions, rng, n_batches=bc_steps, lr=bc_lr
        )
        print(
            f"BC done: loss={metrics['final_loss']:.4f}, "
            f"acc={metrics['final_accuracy']:.4f}"
        )

    callbacks = [
        EvalCallback(
            eval_env,
            best_model_save_path=str(checkpoint_path),
            eval_freq=eval_freq,
            n_eval_episodes=10,
            deterministic=True,
        ),
        CheckpointCallback(
            save_freq=checkpoint_freq,
            save_path=str(checkpoint_path),
            name_prefix="rl_model",
        ),
    ]
    if pareto_reference is not None:
        from ai.agents.warm_start import WarmStartCallback

        callbacks.append(WarmStartCallback())

    print(f"Training {algorithm} for {total_timesteps} timesteps...")
    print(f"  Net arch: {net_arch}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Checkpoints: {checkpoint_path}")
    print(f"  TensorBoard logs: {tb_log_dir}")
    print(f"  Fairness alpha: {fairness_alpha}")
    if pareto_reference is not None:
        print(f"  Pareto reference: {len(pareto_reference)} feasible points")
    if warm_start:
        print(f"  Warm-start: {warm_start} ({warm_start_solutions} solutions, "
              f"{bc_steps} BC steps)")

    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        tb_log_name=algorithm,
    )

    final_path = checkpoint_path / "final_model"
    model.save(str(final_path))

    metadata = {
        "algorithm": algorithm,
        "total_timesteps": total_timesteps,
        "learning_rate": learning_rate,
        "net_arch": net_arch,
        "env_config": {
            "num_employees": config.num_employees,
            "employee_types": config.employee_types,
            "days": config.days,
            "shifts_per_day": config.shifts_per_day,
            "shift_lengths": config.shift_lengths,
            "ft_max_hours": config.ft_max_hours,
            "pt_max_hours": config.pt_max_hours,
            "fairness_alpha": config.fairness_alpha,
            "pareto_reference_size": (
                len(config.pareto_reference) if config.pareto_reference else 0
            ),
        },
        "warm_start": {
            "type": warm_start,
            "n_solutions": warm_start_solutions if warm_start else None,
            "bc_steps": bc_steps if warm_start else None,
            "bc_lr": bc_lr if warm_start else None,
        },
    }
    metadata_path = checkpoint_path / "model_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nTraining complete.")
    print(f"  Final model: {final_path}.zip")
    print(f"  Best model: {checkpoint_path / 'best_model.zip'}")
    print(f"  Metadata: {metadata_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RL scheduling agent with SB3")
    parser.add_argument(
        "--algorithm",
        type=str,
        default="maskable_ppo",
        choices=list(ALGORITHM_MAP.keys()),
    )
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--tb-log-dir", type=str, default="logs")
    parser.add_argument("--eval-freq", type=int, default=5000)
    parser.add_argument("--checkpoint-freq", type=int, default=10000)
    parser.add_argument("--net-arch", type=int, nargs="+", default=[128, 64])
    parser.add_argument(
        "--fairness-alpha",
        type=float,
        default=2.0,
        help="α-fairness parameter for the reward shaping.",
    )
    parser.add_argument(
        "--warm-start",
        choices=["cpsat"],
        default=None,
        help="Run BC on CP-SAT-derived trajectories before PPO training.",
    )
    parser.add_argument(
        "--warm-start-solutions",
        type=int,
        default=20,
        help="Number of CP-SAT-derived schedules for BC (default 20).",
    )
    parser.add_argument(
        "--bc-steps",
        type=int,
        default=5000,
        help="BC SGD batches (default 5000).",
    )
    parser.add_argument(
        "--bc-lr",
        type=float,
        default=1e-3,
        help="BC learning rate (default 1e-3).",
    )
    parser.add_argument(
        "--pareto-ref",
        type=str,
        default=None,
        help="Path to JSON file with CCMO Pareto front for ΔHV shaping.",
    )

    args = parser.parse_args()

    train(
        algorithm=args.algorithm,
        total_timesteps=args.total_timesteps,
        learning_rate=args.lr,
        checkpoint_dir=args.checkpoint_dir,
        tb_log_dir=args.tb_log_dir,
        eval_freq=args.eval_freq,
        checkpoint_freq=args.checkpoint_freq,
        net_arch=args.net_arch,
        fairness_alpha=args.fairness_alpha,
        warm_start=args.warm_start,
        warm_start_solutions=args.warm_start_solutions,
        bc_steps=args.bc_steps,
        bc_lr=args.bc_lr,
        pareto_ref=args.pareto_ref,
    )


if __name__ == "__main__":
    main()
