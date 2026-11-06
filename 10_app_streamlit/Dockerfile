FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim

WORKDIR /streamlit_app

COPY pyproject.toml uv.lock ./

RUN uv sync --locked --no-dev --extra streamlit --no-install-project

COPY README.md ./
COPY .streamlit/ ./.streamlit/
COPY 03_src/ ./03_src
COPY 10_app_streamlit ./10_app_streamlit
COPY 11_configs/ ./11_configs

RUN uv sync --locked --no-dev --extra streamlit

EXPOSE 8501

CMD ["uv","run","--no-sync","streamlit","run","10_app_streamlit/app_streamlit.py","--server.address","0.0.0.0","--server.port","8501"]