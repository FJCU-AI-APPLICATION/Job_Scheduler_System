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

from agents.environment import EnvironmentConfig, SchedulingEnv
from agents.registry import ALGORITHM_MAP


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
) -> None:
    if algorithm not in ALGORITHM_MAP:
        print(f"Unknown algorithm: {algorithm}. Choose from: {list(ALGORITHM_MAP.keys())}")
        sys.exit(1)

    if net_arch is None:
        net_arch = [128, 64]

    config = EnvironmentConfig()
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

    print(f"Training {algorithm} for {total_timesteps} timesteps...")
    print(f"  Net arch: {net_arch}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Checkpoints: {checkpoint_path}")
    print(f"  TensorBoard logs: {tb_log_dir}")

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
    )


if __name__ == "__main__":
    main()
