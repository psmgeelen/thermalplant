FROM python:3.12-slim-bookworm

RUN pip install poetry==1.8.3

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

WORKDIR /app

COPY pyproject.toml poetry.lock ./

# Install system dependencies including those needed for GPIO
RUN apt update && apt install -y \
        libasound2 \
        alsa-utils \
        gcc \
        libportaudio2 \
        portaudio19-dev \
        curl\
        pipewire pipewire-audio-client-libraries libspa-0.2-modules pipewire-bin


## AI Assistant



    
RUN poetry install --without dev --no-root

# Copy application code after dependencies are installed
COPY app/ /app


CMD ["poetry", "run", "python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8123", "--reload"]