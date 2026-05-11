"""Training CLI for LAST-RL (issue #19).

Two subcommands:
  train-ours  — train on the default EnvironmentConfig (our domain)
  train-paper — sanity-check on a LAST-RL paper benchmark instance
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from ai.agents.environment import EnvironmentConfig
from ai.domain.problem import SchedulingProblem
from ai.domain.schemas import LastRLConfigSnapshot
from ai.optimizers.last_rl import SARSALambdaPolicy, save_policy, train
from ai.optimizers.last_rl_problem import RosteringLastRLProblem
from ai.optimizers.result import LastRLConfig, LastRLEpisodeStatus


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--num-episodes", type=int, default=200)
    p.add_argument("--episode-length", type=int, default=500)
    p.add_argument("--wall-clock-budget-s", type=float, default=None)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.9)
    p.add_argument("--epsilon-start", type=float, default=0.5)
    p.add_argument("--epsilon-end", type=float, default=0.05)
    p.add_argument("--iht-size", type=int, default=4096)
    p.add_argument("--num-tilings", type=int, default=8)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--output-dir", default="checkpoints/last_rl")
    p.add_argument(
        "--save-trajectory",
        action="store_true",
        help="Also write a <name>.episodes.json sibling containing per-episode telemetry",
    )
    p.add_argument("--verbose", action="store_true")


def _config_from_args(args) -> LastRLConfig:
    return LastRLConfig(
        num_episodes=args.num_episodes,
        episode_length=args.episode_length,
        wall_clock_budget_s=args.wall_clock_budget_s,
        alpha=args.alpha,
        gamma=args.gamma,
        lam=args.lam,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        iht_size=args.iht_size,
        num_tilings=args.num_tilings,
        ip_time_budget_s=getattr(args, "ip_time_budget_s", 2.0),
        ip_workers=getattr(args, "ip_workers", 2),
        seed=args.seed,
    )


def _snapshot_from(sp: SchedulingProblem, config: LastRLConfig) -> LastRLConfigSnapshot:
    return LastRLConfigSnapshot(
        num_employees=sp.num_employees,
        employee_types=list(sp.employee_types),
        days=sp.days,
        shifts_per_day=sp.shifts_per_day,
        shift_lengths=list(sp.shift_lengths),
        num_episodes=config.num_episodes,
        episode_length=config.episode_length,
        wall_clock_budget_s=config.wall_clock_budget_s,
        alpha=config.alpha,
        gamma=config.gamma,
        lam=config.lam,
        epsilon_start=config.epsilon_start,
        epsilon_end=config.epsilon_end,
        iht_size=config.iht_size,
        num_tilings=config.num_tilings,
        ip_time_budget_s=config.ip_time_budget_s,
        ip_workers=config.ip_workers,
        fairness_alpha=config.fairness_alpha,
        seed=config.seed,
    )


def _ep_to_status(ep_idx: int, epsilon: float, ep) -> LastRLEpisodeStatus:
    rewards = [s.reward for s in ep.step_history]
    improving = sum(1 for r in rewards if r > 0)
    return LastRLEpisodeStatus(
        episode=ep_idx,
        epsilon=epsilon,
        initial_cost=ep.initial_cost,
        final_cost=ep.final_cost,
        best_cost_in_episode=ep.best_cost,
        neighborhood_usage=ep.neighborhood_usage,
        wall_clock_s=ep.wall_clock_s,
        total_reward=sum(rewards),
        mean_step_reward=(sum(rewards) / len(rewards)) if rewards else 0.0,
        fraction_improving_steps=(improving / len(rewards)) if rewards else 0.0,
    )


def cmd_train_ours(args) -> int:
    config = _config_from_args(args)
    env = EnvironmentConfig()
    problem = SchedulingProblem.from_config(env)
    last_rl_problem = RosteringLastRLProblem(problem, config)
    policy = SARSALambdaPolicy(
        iht_size=config.iht_size,
        num_tilings=config.num_tilings,
        num_actions=last_rl_problem.num_actions,
    )

    rng = np.random.default_rng(config.seed)
    epsilon_schedule = np.linspace(
        config.epsilon_start, config.epsilon_end, config.num_episodes
    )
    episodes = train(last_rl_problem, policy, config, rng)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "last_rl_policy.npz"
    snapshot = _snapshot_from(problem, config)
    # NB: use stdlib json.dumps (not pydantic's model_dump_json) so that
    # float('inf') round-trips as JSON Infinity instead of null.
    save_policy(
        policy, ckpt,
        config_snapshot_json=json.dumps(snapshot.model_dump()),
    )
    print(f"Wrote {ckpt}")
    print(
        f"  num_episodes={config.num_episodes} "
        f"best_cost={min(ep.best_cost for ep in episodes):.2f} "
        f"final_epsilon={float(epsilon_schedule[-1]):.3f}"
    )

    if args.save_trajectory:
        traj_path = out_dir / "last_rl_policy.episodes.json"
        statuses = [
            _ep_to_status(i, float(epsilon_schedule[i]), ep)
            for i, ep in enumerate(episodes)
        ]
        traj_path.write_text(json.dumps([s.model_dump() for s in statuses], indent=2))
        print(f"Wrote {traj_path}")
    return 0


def cmd_train_paper(args) -> int:
    try:
        from ai.data.last_rl_benchmark import load_paper_instance
        from ai.optimizers.last_rl_problem import PaperBenchmarkProblem
    except ImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    path = Path(args.instance_path)
    if not path.exists():
        print(
            f"error: instance file not found: {args.instance_path}",
            file=sys.stderr,
        )
        return 1
    try:
        instance = load_paper_instance(str(path))
    except (NotImplementedError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    problem = PaperBenchmarkProblem(instance)
    config = _config_from_args(args)
    policy = SARSALambdaPolicy(
        iht_size=config.iht_size,
        num_tilings=config.num_tilings,
        num_actions=problem.num_actions,
    )
    rng = np.random.default_rng(config.seed)
    episodes = train(problem, policy, config, rng)

    achieved = min(ep.best_cost for ep in episodes)
    target = instance.paper_target_cost
    within_10pct = abs(achieved - target) / max(target, 1.0) <= 0.10

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / f"last_rl_paper_{instance.name}.npz"
    save_policy(
        policy, ckpt,
        config_snapshot_json=json.dumps({
            "paper_instance": instance.name,
            "achieved_cost": achieved,
            "paper_target_cost": target,
            "within_10pct": within_10pct,
            "config": config.model_dump(),
        }),
    )
    print(f"Wrote {ckpt}")
    print(
        f"  paper_target={target:.1f} achieved={achieved:.1f} "
        f"within_10pct={within_10pct} {'✓' if within_10pct else '✗'}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train LAST-RL (issue #19)."
    )
    subs = parser.add_subparsers(dest="cmd", required=True)

    p_ours = subs.add_parser(
        "train-ours",
        help="Train on the default EnvironmentConfig (our SchedulingProblem).",
    )
    _add_common_args(p_ours)
    p_ours.add_argument("--ip-time-budget-s", type=float, default=2.0)
    p_ours.add_argument("--ip-workers", type=int, default=2)
    p_ours.set_defaults(func=cmd_train_ours)

    p_paper = subs.add_parser(
        "train-paper",
        help="Sanity check on a LAST-RL paper benchmark.",
    )
    _add_common_args(p_paper)
    p_paper.add_argument(
        "--instance-path", required=True,
        help="Path to BCV-format text file from Kletzander's GitLab.",
    )
    p_paper.set_defaults(func=cmd_train_paper)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
