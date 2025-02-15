# https://github.com/ptrsr/pi-ci
FROM ptrsr/pi-ci

CMD docker.io docker-compose-v2

RUN mkdir thermalplant
WORKDIR thermalplant
COPY /services ./services
COPY .env ./env
COPY docker-compose.yaml ./docker-compose.yaml

ENTRYPOINT ["docker", "compose", "up"]

### TODO, make workflow,
# 1. resize image
# 2. Install dependencies
# 3. mount shizzle for project
# 4. start docker-compose.