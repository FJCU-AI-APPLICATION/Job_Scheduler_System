# Job Scheduler System

Multi-service shift-scheduling platform.

| Service | Stack | URL |
|---|---|---|
| `frontend` | Gradio | http://localhost:8080 |
| `backend` | FastAPI + SQLAlchemy + Alembic | http://localhost:8002 (Swagger at `/docs`) |
| `ai-server` | FastAPI + Stable-Baselines3 + EvoTorch + OR-Tools | http://localhost:8003 (Swagger at `/docs`) |
| `database` | Postgres 16 | `localhost:5432` |

## Quick start

```bash
docker-compose up --build
```

Then open http://localhost:8080.

## Repo layout

```
src/backend/    FastAPI subpackage   (api, core, db, domain, schemas, services)
src/ai/         AI subpackage        (api, core, domain, agents, optimizers,
                                      services, training, data)
src/frontend/   Gradio subpackage    (api_client, core, views)
alembic/        DB migrations (backend)
envs/           per-env config (dev.env, ...)
data/           training datasets
notebooks/      research notebooks
checkpoints/    trained model artifacts
pyproject.toml  one project; deps split via [project.optional-dependencies] {ai,backend,frontend}
Dockerfile      one image; docker-compose runs three containers with different commands
docker-compose.yml
```

## Optimizers

The AI service ships three optimizers — two evolutionary and one exact — all registered on the `Optimizer` ABC and selectable by name:

| Algorithm | Module | Notes |
|---|---|---|
| `nsga2` | `ai.optimizers.nsga2.NSGAIIOptimizer` | EvoTorch `GeneticAlgorithm` with day-aligned crossover, uniform-int mutation, and a vectorized repair operator |
| `ccmo` | `ai.optimizers.ccmo.CCMOOptimizer` | Constrained MOEA via Coevolution (Tian et al. 2021); Pop1 selects under Deb's constraint-domination, Pop2 explores unconstrained |
| `cpsat` | `ai.optimizers.cpsat.CPSATOptimizer` | Exact baseline via OR-Tools CP-SAT; lexicographic two-stage (minimize back-to-back, then minimize max-min fairness gap); single optimal schedule per run |
| `matheuristic` | `ai.optimizers.matheuristic.MatheuristicOptimizer` | Hybrid IP + VNS / SA matheuristic (Burke et al. EJOR 2017; Ceschia et al. Annals OR 2017); CP-SAT inner-IP slices warm-started from the incumbent; single schedule per run |

Inference: `POST /predict/evolutionary/{algorithm}` — e.g. `/predict/evolutionary/nsga2`. The legacy `POST /predict/ga` is **deprecated** and forwards to `nsga2`. For exact ground-truth schedules at the default size, use `POST /predict/cpsat` (see the CP-SAT row above).

### Training

```bash
python -m ai.training.evolutionary --algorithm nsga2 --generations 200 --pop-size 100
python -m ai.training.evolutionary --algorithm ccmo  --generations 200 --pop-size 100 --device cuda
```

For an exact baseline:

```bash
python -m ai.training.cpsat                                       # default 30s/stage budget
python -m ai.training.cpsat --timeout-s-per-stage 60 --seed 42    # tighter run
```

For the IP + VNS / SA matheuristic (#18):

```bash
python -m ai.training.matheuristic --time-budget-s 300 --seed 42
python -m ai.training.matheuristic --acceptance sa --sa-initial-temperature 50 --seed 42
```

For RL with CP-SAT warm-start + Pareto-shaped reward:

```bash
# 1. Produce a Pareto front via CCMO
python -m ai.training.evolutionary --algorithm ccmo --generations 200 --pop-size 100 --output-dir checkpoints

# 2. Train MaskablePPO with warm-start + ΔHV reward
python -m ai.training.rl \
    --algorithm maskable_ppo \
    --warm-start cpsat \
    --pareto-ref checkpoints/ccmo_pareto_front.json \
    --total-timesteps 500000
```

### Benchmark

See [`BENCHMARKS.md`](BENCHMARKS.md) for the INRC-I sprint-track A/B harness.

## Documentation

Full documentation lives in the **[Wiki](https://github.com/FJCU-AI-APPLICATION/Job_Scheduler_System/wiki)** — one page per `src/` sub-module, plus:

- [Architecture](https://github.com/FJCU-AI-APPLICATION/Job_Scheduler_System/wiki/Architecture) — service boundaries, request flow, deployment topology
- [Local Development](https://github.com/FJCU-AI-APPLICATION/Job_Scheduler_System/wiki/Local-Development) — `uv sync` workflow, migrations, training, single-service runs
- [Database Design](https://github.com/FJCU-AI-APPLICATION/Job_Scheduler_System/wiki/Database-Design) — entities, relationships, constraints

## Contributing

PRs go to `main`. See the wiki's [Local Development](https://github.com/FJCU-AI-APPLICATION/Job_Scheduler_System/wiki/Local-Development) page for the per-service `uv sync` workflow and migration commands.
