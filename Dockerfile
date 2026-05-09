FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --all-extras

COPY . .

CMD ["uv", "run", "uvicorn", "backend.main:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
