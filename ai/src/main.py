from fastapi import FastAPI

from api import health, inference

app = FastAPI(title="Job Scheduler AI Service", version="1.0.0")

app.include_router(health.router)
app.include_router(inference.router)
