# Use Python 3.11 slim image optimized for free tier
FROM python:3.11-slim

# Set environment variables for free tier optimization
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8080 \
    WORKERS=1 \
    MAX_WORKERS=1 \
    WEB_CONCURRENCY=1 \
    TIMEOUT=300

# Set work directory
WORKDIR /app

# Install minimal system dependencies for free tier (no TA-Lib to avoid build issues)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first for better Docker layer caching
COPY requirements_free_tier.txt requirements.txt

# Install Python dependencies with memory optimization
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --no-deps -r requirements.txt

# Copy application code
COPY financial_mcp_server_free_tier.py financial_mcp_server.py
COPY config/ config/
COPY utils/ utils/
COPY models/ models/
COPY analysis/ analysis/

# Create necessary directories with minimal footprint
RUN mkdir -p models/cache logs

# Expose port
EXPOSE $PORT

# Health check optimized for free tier
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=2 \
    CMD curl -f http://localhost:$PORT/health || exit 1

# Run with single worker for free tier
CMD ["python", "-m", "uvicorn", "financial_mcp_server:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1", "--timeout-keep-alive", "300"]
