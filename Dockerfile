FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv==0.12.7

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY pollicount ./pollicount
RUN uv sync --frozen --no-dev

ENV PATH=/app/.venv/bin:$PATH \
    POLLICOUNT_DATA=/data
RUN useradd --create-home app && mkdir /data && chown app /data
USER app

EXPOSE 8000
CMD ["uvicorn", "--factory", "pollicount.server:production_app", "--host", "0.0.0.0", "--port", "8000"]
