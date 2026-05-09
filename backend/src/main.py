from contextlib import asynccontextmanager

from fastapi import FastAPI

from api import employee, policy, schedule
from db.session import engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await engine.dispose()


app = FastAPI(
    title="Job Scheduler API",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(employee.router)
app.include_router(schedule.router)
app.include_router(policy.router)


@app.get("/health")
async def health():
    return {"status": "ok"}
