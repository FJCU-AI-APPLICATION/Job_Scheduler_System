# Benchmarks

## INRC-I sprint track — A/B between NSGA-II and CCMO

The benchmark harness compares `NSGAIIOptimizer` and `CCMOOptimizer` on the sprint track of the **First International Nurse Rostering Competition** (Haspeslagh et al. 2014, *Annals of Operations Research* 218(1)).

### What this is and isn't

- **What it is:** an internal A/B harness on real-shaped instances, scored by hypervolume on the feasible Pareto front and tested for significance by Wilcoxon signed-rank.
- **What it isn't:** a comparison against the published INRC-I leaderboard. Our `SchedulingProblem` is a simpler model than full INRC-I — the lossy adapter drops nurse skills, contract patterns, weekend rotation, shift sequencing, and the original soft-constraint weights. The hypervolume scores reported here are **not** INRC-I aggregate competition scores.

### Setup

```bash
python scripts/fetch_inrc1.py    # one-time download to data/benchmarks/inrc1/
```

The script pulls the official PATAT mirror's `instances.zip`, extracts the nested sprint archive, and writes `sprint01.xml` … `sprint10.xml`. The download (~250 KB) is gitignored.

### Running

```bash
# Single algorithm
python -m ai.training.benchmark --algorithm nsga2 --track sprint --seeds 10 --report nsga2.json
python -m ai.training.benchmark --algorithm ccmo  --track sprint --seeds 10 --report ccmo.json

# A/B
python -m ai.training.benchmark --algorithm nsga2,ccmo --track sprint --seeds 10 --report a_b.json
```

### What gets reported

- **Per run:** `instance`, `algorithm`, `seed`, `hypervolume`, `feasible_front_size`, best `unfairness` / `violations` / `b2b`, `wall_clock_s`.
- **Aggregate (per instance):** HV mean and std for each algorithm, plus a Wilcoxon signed-rank p-value when both algorithms have ≥ 6 paired non-equal seeds; otherwise `wilcoxon_p` is `null`.

### Reference point

Hypervolume is computed against the reference point `(unfairness, violations, b2b) = (2.0, 1000.0, 100.0)`, which dominates any plausible objective tuple on the sprint track. The `unfairness` ceiling is `2.0` (bumped from `1.0`) to cover adversarial α=1 (Nash) cases where unfairness can exceed 1. Tighten this if expanding to medium or long tracks.

### Hypervolume on the feasible front only

We filter `pareto_fitnesses` to entries with `violations == 0` before computing HV. This prevents an algorithm from inflating HV by widening its infeasible Pareto front — a real risk with NSGA-II's three-objective ranking, which treats `violations` as just another objective rather than a constraint.
