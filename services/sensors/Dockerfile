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
        build-essential \
        python3-dev \
        libasound2 \
        libasound2-dev \
        portaudio19-dev \
        alsa-utils \
        gcc \
        libportaudio2 \
        portaudio19-dev \
        curl
    
RUN apt-get install pkg-config libxml2-dev libxmlsec1-dev libxmlsec1-openssl -y

RUN poetry install --without dev --no-root

# Copy application code after dependencies are installed
COPY app/ /app


CMD ["poetry", "run", "python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8123", "--reload"]