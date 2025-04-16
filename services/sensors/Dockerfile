FROM python:3.12-slim-bookworm

RUN pip install poetry==1.8.3

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

WORKDIR /app
COPY app/ /app
COPY pyproject.toml poetry.lock ./

RUN apt update && apt install \
        gcc \
        libportaudio2 \
        portaudio19-dev -y

RUN poetry install --without dev --no-root

CMD ["poetry", "run", "python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8123", "--reload"]
