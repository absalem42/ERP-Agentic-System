FROM python:slim-trixie

WORKDIR /app

RUN apt-get update && apt-get install -y \
    bash \
    curl \
    nginx \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

COPY erp_system/requirements.txt .

RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    /root/.local/bin/uv pip install --system --no-cache-dir --only-binary=all -r requirements.txt

COPY erp_system/ .

RUN mkdir -p databases logs && \
    chmod +x frontend/start_streamlit.sh && \
    chmod +x docker/start-app.sh

ENV PYTHONPATH=/app/backend
ENV DB_PATH=/app/databases/erp.db

EXPOSE 7860 8000 8501

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["bash", "docker/start-app.sh"]
