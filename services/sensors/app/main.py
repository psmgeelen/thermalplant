import os
from fastapi import FastAPI, Request
from fastapi.openapi.utils import get_openapi
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi_health import health
from sensors import TempSensor
import logging

# Setup Logger
logging.basicConfig(
    format="%(levelname)s - %(asctime)s - %(name)s - %(message)s", level=logging.WARNING
)
logger = logging.getLogger("my-logger")
stream_handler = logging.StreamHandler()
logger.addHandler(stream_handler)


# Init API
limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)



def my_schema():
    DOCS_TITLE = "ThermalPlant Sensors"
    DOCS_VERSION = "0.1"
    openapi_schema = get_openapi(
        title=DOCS_TITLE,
        version=DOCS_VERSION,
        routes=app.routes,
    )
    openapi_schema["info"] = {
        "title": DOCS_TITLE,
        "version": DOCS_VERSION,
        "description": "A service that delivers free to use, truly random numbers",
        "contact": {
            "name": "A project of geelen.io",
            "url": "https://github.com/psmgeelen/thermalplant",
        },
        "license": {"name": "MIT", "url": "http://opensource.org/license/mit/"},
    }
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = my_schema


##### Endpoints #####
@app.get(
    "/ping",
    summary="Check whether there is a connection at all",
    description="You ping, API should Pong",
    response_description="A string saying Pong",
)
def ping():
    return "pong"


@app.get(
    "/temperature_upper",
    summary="Get the temperature of the upper Segment.",
    description= "",
    response_description="A dictionary with a list of devices",
    response_model=str,
)
@limiter.limit("500/minute")
def get_temperature_upper(request: Request,):
    sensor = TempSensor(spi_port=1, chip_select=1)
    temp = sensor.read_temperature()
    sensor.close()
    return temp


@app.get(
    "/temperature_lower",
    summary="Get a list of all the available devices",
    description=(
        "This request returns a list of devices. If no hardware is found, it will"
        "return the definition of the DeviceEmulator class"
    ),
    response_description="A dictionary with a list of devices",
    response_model=str,
)
@limiter.limit("500/minute")
def get_temperature_lower(request: Request,):
    sensor = TempSensor(spi_port=1, chip_select=0)
    temp = sensor.read_temperature()
    sensor.close()
    return temp


##### Healthchecks #####
def _healthcheck_ping():
    hostname = "google.com"  # example
    response = os.system("ping -c 1 " + hostname)
    # and then check the response...
    if response == 0:
        return str(response)
    else:
        return False

def _healthcheck_spi():
    sensor = TempSensor(spi_port=1, chip_select=0)
    temp = sensor.read_temperature()
    sensor.close()
    if type(temp) == float:
        return str(temp)
    else:
        return False


app.add_api_route(
    "/health",
    health(
        [
            _healthcheck_ping,
            _healthcheck_spi
        ]
    ),
    summary="Check the health of the service",
    description=(
        "The healthcheck not only checks whether the service is up, but it will also"
        " check for internet connectivity, whether the hardware is callable and it does"
        " an end-to-end test. The healthcheck therefore can become blocking by nature."
        " Use with caution!"
    ),
    response_description=(
        "The response is only focused around the status. 200 is OK, anything else and"
        " there is trouble."
    ),
)