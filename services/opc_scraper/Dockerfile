FROM ghcr.io/astral-sh/uv:python3.12-alpine

WORKDIR /app

# Copy project config and code
COPY pyproject.toml ./
COPY app/ .

# Install deps (using PEP 621 project layout)
RUN apk add --no-cache build-base && \
    uv sync --no-cache \
    && apk del build-base

# Run the app
ENTRYPOINT ["uv", "run", "python", "main.py"]

