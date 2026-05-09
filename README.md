# Job Scheduler System

Multi-service shift-scheduling platform.

| Service | Stack | URL |
|---|---|---|
| `frontend` | Gradio | http://localhost:8080 |
| `backend` | FastAPI + SQLAlchemy + Alembic | http://localhost:8002 (Swagger at `/docs`) |
| `ai-server` | FastAPI + Stable-Baselines3 + DEAP | http://localhost:8003 (Swagger at `/docs`) |
| `database` | Postgres 16 | `localhost:5432` |

## Quick start

```bash
docker-compose up --build
```

Then open http://localhost:8080.

## Repo layout

```
backend/    FastAPI service — flat src/ modules (api, core, db, domain, schemas, services)
ai/         AI service       — flat src/ modules (api, core, domain, agents, optimizers,
                                                  services, training, data)
frontend/   Gradio service   — flat src/ modules (api_client, core, views)
docker-compose.yml
```

## Documentation

Full documentation lives in the **[Wiki](https://github.com/FJCU-AI-APPLICATION/Job_Scheduler_System/wiki)** — one page per `src/` sub-module, plus:

- [Architecture](https://github.com/FJCU-AI-APPLICATION/Job_Scheduler_System/wiki/Architecture) — service boundaries, request flow, deployment topology
- [Local Development](https://github.com/FJCU-AI-APPLICATION/Job_Scheduler_System/wiki/Local-Development) — `uv sync` workflow, migrations, training, single-service runs
- [Database Design](https://github.com/FJCU-AI-APPLICATION/Job_Scheduler_System/wiki/Database-Design) — entities, relationships, constraints

## Contributing

PRs go to `main`. See the wiki's [Local Development](https://github.com/FJCU-AI-APPLICATION/Job_Scheduler_System/wiki/Local-Development) page for the per-service `uv sync` workflow and migration commands.
