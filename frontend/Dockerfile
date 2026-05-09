FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV PYTHONPATH=/usr/src/src

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /usr/src

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY . .

WORKDIR /usr/src/src

EXPOSE 7860
CMD ["uv", "run", "python", "main.py"]
