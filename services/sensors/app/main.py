import os
from fastapi import FastAPI, Request, Body, HTTPException
from fastapi.openapi.utils import get_openapi
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi_health import health
from sensors import TempSensor, RPMSensor, RPMSensorSettings, AudioHandler, AudioHandlerSettings
import logging

# Setup Logger
logging.basicConfig(
    format="%(levelname)s - %(asctime)s - %(name)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger("API")
stream_handler = logging.StreamHandler()
logger.addHandler(stream_handler)


# Init API
limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Fixed sensor parameters
GPIO_PIN = 22
AUDIO_RATE = 44100
AUDIO_CHANNELS = 1

# Configurable sensor parameters with defaults
rpm_settings = {
    "measurement_window": 100,  # measures over 1 second the rpm
    "measurement_interval": 0.001,  # 1000 times /sec
    "sample_size": 8  # How many sample do we need to take in order to make sure that we are not skipping a cycle
}

audio_settings = {
    "sample_duration": 1.0,
    "mfcc_count": 50,
    "buffer_size": 3
}

# Initialize sensors with current settings
def initialize_rpm_sensor():
    global rpm_sensor
    try:
        # If the sensor exists, close/stop it first
        if 'rpm_sensor' in globals():
            try:
                rpm_sensor.stop()
            except:
                pass
                
        rpm_sensor = RPMSensor(
            gpio_pin=GPIO_PIN,
            measurement_window=rpm_settings["measurement_window"],
            measurement_interval=rpm_settings["measurement_interval"],
            sample_size=rpm_settings["sample_size"]
        )
        logger.info(f"RPM sensor initialized with settings: {rpm_settings}")
        return rpm_sensor
    except Exception as e:
        logger.error(f"Error initializing RPM sensor: {e}")
        raise

def initialize_audio_handler():
    global audio_sensor
    try:
        # If the sensor exists, close it first
        if 'audio_sensor' in globals():
            try:
                audio_sensor.close()
            except:
                pass
                
        audio_sensor = AudioHandler(
            rate=AUDIO_RATE,
            channels=AUDIO_CHANNELS,
            sample_duration=audio_settings["sample_duration"],
            mfcc_count=audio_settings["mfcc_count"],
            buffer_size=audio_settings["buffer_size"]
        )
        logger.info(f"Audio handler initialized with settings: {audio_settings}")
        return audio_sensor
    except Exception as e:
        logger.error(f"Error initializing audio handler: {e}")
        raise

# Activate Sensors
rpm_sensor = initialize_rpm_sensor()
audio_sensor = initialize_audio_handler()


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

@app.get(
    "/rpm",
    summary="Get RPMs of Fan",
    description=(
        "This request returns the current RPM reading from the fan sensor"
    ),
    response_description="Current RPM value",
    response_model=float,
)
@limiter.limit("500/minute")
def get_rpm(request: Request):
    return rpm_sensor.read_rpm()

@app.get(
    "/rpm/settings",
    summary="Get RPM sensor settings",
    description="Returns the current configuration settings for the RPM sensor",
    response_description="Dictionary containing the RPM sensor settings",
    response_model=RPMSensorSettings,
)
@limiter.limit("100/minute")
def get_rpm_settings(request: Request):
    return RPMSensorSettings(**rpm_settings)

@app.put(
    "/rpm/settings",
    summary="Update RPM sensor settings",
    description="Update the configuration settings for the RPM sensor and reinitialize it",
    response_description="Dictionary containing the updated RPM sensor settings",
    response_model=RPMSensorSettings,
)
@limiter.limit("20/minute")
def update_rpm_settings(request: Request, settings: RPMSensorSettings):
    try:
        # Update the global settings
        global rpm_settings
        rpm_settings = settings.dict()
        
        # Reinitialize the sensor with new settings
        initialize_rpm_sensor()
        
        return settings
    except Exception as e:
        logger.error(f"Failed to update RPM sensor settings: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update RPM sensor settings: {str(e)}")


@app.get(
    "/mfcc",
    summary="Get MFCC (Sound) from Engine",
    description=(
        "Returns MFCC coefficients with frequency labels in a dictionary format"
    ),
    response_description="A dictionary of labeled MFCC coefficients",
)
@limiter.limit("500/minute")
def get_mfcc(request: Request):
    return audio_sensor.read_mfcc()

@app.get(
    "/audio/settings",
    summary="Get audio handler settings",
    description="Returns the current configuration settings for the audio handler",
    response_description="Dictionary containing the audio handler settings",
    response_model=AudioHandlerSettings,
)
@limiter.limit("100/minute")
def get_audio_settings(request: Request):
    return AudioHandlerSettings(**audio_settings)

@app.put(
    "/audio/settings",
    summary="Update audio handler settings",
    description="Update the configuration settings for the audio handler and reinitialize it",
    response_description="Dictionary containing the updated audio handler settings",
    response_model=AudioHandlerSettings,
)
@limiter.limit("20/minute")
def update_audio_settings(request: Request, settings: AudioHandlerSettings):
    try:
        # Update the global settings
        global audio_settings
        audio_settings = settings.dict()
        
        # Reinitialize the audio handler with new settings
        initialize_audio_handler()
        
        return settings
    except Exception as e:
        logger.error(f"Failed to update audio handler settings: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update audio handler settings: {str(e)}")


@app.get(
    "/spectrum",
    summary="Get Spectrum (Sound) from Engine",
    description=(
        "Returns frequency spectrum data with labeled frequency bands"
    ),
    response_description="A dictionary of labeled frequency bands",
)
@limiter.limit("500/minute")
def get_spectrum(request: Request):
    return audio_sensor.read_spectrum()


@app.get(
    "/audio",
    summary="Get all audio data",
    description=(
        "Returns both MFCC and spectrum data with frequency labels"
    ),
    response_description="A dictionary containing both MFCC and spectrum data",
)
@limiter.limit("500/minute")
def get_audio(request: Request):
    return audio_sensor.read_all_audio()


@app.get(
    "/sensors",
    summary="Get all sensor data",
    description=(
        "Returns all sensor data including temperature, RPM, and audio features"
    ),
    response_description="A dictionary containing all sensor readings with appropriate labels",
)
@limiter.limit("200/minute")
def get_all_sensors(request: Request):
    # Get temperature readings
    temp_sensor_upper = TempSensor(spi_port=1, chip_select=1)
    temp_upper = temp_sensor_upper.read_temperature()
    temp_sensor_upper.close()
    
    temp_sensor_lower = TempSensor(spi_port=1, chip_select=0)
    temp_lower = temp_sensor_lower.read_temperature()
    temp_sensor_lower.close()
    
    # Get RPM reading
    rpm = rpm_sensor.read_rpm()
    
    # Get audio features
    audio_data = audio_sensor.read_all_audio()
    
    # Combine all data
    all_sensors = {
        "temperature_upper": temp_upper,
        "temperature_lower": temp_lower,
        "rpm": rpm,
        "audio": audio_data
    }
    
    return all_sensors

@app.get(
    "/settings",
    summary="Get all sensor settings",
    description="Returns the current configuration settings for all sensors",
    response_description="Dictionary containing all sensor settings",
)
@limiter.limit("100/minute")
def get_all_settings(request: Request):
    return {
        "rpm": rpm_settings,
        "audio": audio_settings,
        "fixed_settings": {
            "rpm": {
                "gpio_pin": GPIO_PIN
            },
            "audio": {
                "rate": AUDIO_RATE,
                "channels": AUDIO_CHANNELS
            }
        }
    }



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