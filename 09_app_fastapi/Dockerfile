FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim

WORKDIR /fastapi_app

COPY pyproject.toml uv.lock ./

RUN uv sync --locked --no-dev --no-install-project

COPY README.md ./
COPY 03_src/ ./03_src
COPY 06_models/ ./06_models
COPY 07_artifacts/ ./07_artifacts
COPY 01_data/04_test_data/ ./01_data/04_test_data/
COPY 09_app_fastapi ./09_app_fastapi
COPY 11_configs/ ./11_configs

RUN uv sync --locked --no-dev

EXPOSE 8080

CMD ["uv","run","--no-sync","uvicorn","09_app_fastapi.app_fastapi:app","--host","0.0.0.0","--port","8080"]
