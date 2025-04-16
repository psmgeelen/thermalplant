FROM python:3.12-alpine

RUN pip install poetry==1.8.3

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

WORKDIR /app
COPY app/ ./
COPY pyproject.toml poetry.lock ./

RUN apk update && apk add python3-dev \
        gcc \
        libc-dev

RUN poetry install --without dev --no-root

ENTRYPOINT ["poetry", "run", "python", "main.py"]