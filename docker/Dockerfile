FROM python:3.11.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1
WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src/ ./src/
COPY web/ ./web/

RUN pip install --no-cache-dir "torch==2.6.0+cpu" --extra-index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -e .

EXPOSE 8000
CMD ["python", "-m", "detector", "serve", "--host", "0.0.0.0"]

