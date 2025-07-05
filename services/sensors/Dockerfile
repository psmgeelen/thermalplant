FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

# Copy project config and code
COPY pyproject.toml ./
COPY app/ .

# Install system dependencies including those needed for GPIO
RUN apt update && apt install -y \
        libasound2 \
        alsa-utils \
        gcc \
        libportaudio2 \
        portaudio19-dev \
        curl\
        pipewire pipewire-audio-client-libraries libspa-0.2-modules pipewire-bin && \
        uv sync --extra cpu --no-cache && \
        rm -rf /var/lib/apt/lists/*

CMD ["uv", "run", "python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8123", "--reload"]