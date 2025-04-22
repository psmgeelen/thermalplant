import os
from fastapi import FastAPI, Request, HTTPException
from fastapi.openapi.utils import get_openapi
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi_health import health
from sensors import (
    TempSensor,
    RPMSensor,
    RPMSensorSettings,
    AudioHandler,
    AudioHandlerSettings,
)
import logging
import math
import psutil

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

audio_settings = {"sample_duration": 1.0, "mfcc_count": 50, "buffer_size": 3}


# Initialize sensors with current settings
def initialize_rpm_sensor():
    global rpm_sensor
    try:
        # If the sensor exists, close/stop it first
        if "rpm_sensor" in globals():
            try:
                rpm_sensor.stop()
            except:
                pass

        rpm_sensor = RPMSensor(
            gpio_pin=GPIO_PIN,
            measurement_window=rpm_settings["measurement_window"],
            measurement_interval=rpm_settings["measurement_interval"],
            sample_size=rpm_settings["sample_size"],
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
        if "audio_sensor" in globals():
            try:
                audio_sensor.close()
            except:
                pass

        audio_sensor = AudioHandler(
            rate=AUDIO_RATE,
            channels=AUDIO_CHANNELS,
            sample_duration=audio_settings["sample_duration"],
            mfcc_count=audio_settings["mfcc_count"],
            buffer_size=audio_settings["buffer_size"],
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
    summary="Basic connectivity test endpoint for API health verification",
    description="You ping, API should Pong",
    response_description="A string saying Pong",
)
def ping():
    return "pong"


@app.get(
    "/temperature_upper",
    summary="Retrieve temperature readings from the upper thermal segment sensor",
    description="",
    response_description="A dictionary with a list of devices",
    response_model=str,
)
@limiter.limit("500/minute")
def get_temperature_upper(
    request: Request,
):
    sensor = TempSensor(spi_port=1, chip_select=1)
    temp = sensor.read_temperature()
    sensor.close()
    return temp


@app.get(
    "/temperature_lower",
    summary="Retrieve temperature readings from the lower thermal segment sensor",
    description=(
        "This request returns a list of devices. If no hardware is found, it will"
        "return the definition of the DeviceEmulator class"
    ),
    response_description="A dictionary with a list of devices",
    response_model=str,
)
@limiter.limit("500/minute")
def get_temperature_lower(
    request: Request,
):
    sensor = TempSensor(spi_port=1, chip_select=0)
    temp = sensor.read_temperature()
    sensor.close()
    return temp


@app.get(
    "/rpm",
    summary="Retrieve real-time rotational speed (RPM) measurement from the fan sensor",
    description=("This request returns the current RPM reading from the fan sensor"),
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
        raise HTTPException(
            status_code=500, detail=f"Failed to update RPM sensor settings: {str(e)}"
        )


@app.get(
    "/mfcc",
    summary="Extract Mel-Frequency Cepstral Coefficients (MFCC) acoustic features from engine sound",
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
        raise HTTPException(
            status_code=500, detail=f"Failed to update audio handler settings: {str(e)}"
        )


@app.get(
    "/spectrum",
    summary="Retrieve frequency-domain acoustic spectrum analysis from engine sound",
    description=("Returns frequency spectrum data with labeled frequency bands"),
    response_description="A dictionary of labeled frequency bands",
)
@limiter.limit("500/minute")
def get_spectrum(request: Request):
    return audio_sensor.read_spectrum()


@app.get(
    "/audio",
    summary="Get all audio data",
    description=("Returns both MFCC and spectrum data with frequency labels"),
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
        "audio": audio_data,
    }

    return all_sensors


@app.get(
    "/settings",
    summary="Retrieve comprehensive configuration parameters for all sensor subsystems",
    description="Returns the current configuration settings for all sensors",
    response_description="Dictionary containing all sensor settings",
)
@limiter.limit("100/minute")
def get_all_settings(request: Request):
    return {
        "rpm": rpm_settings,
        "audio": audio_settings,
        "fixed_settings": {
            "rpm": {"gpio_pin": GPIO_PIN},
            "audio": {"rate": AUDIO_RATE, "channels": AUDIO_CHANNELS},
        },
    }


##### Healthchecks #####
def _healthcheck_ping():
    """
    Verifies external network connectivity by pinging a reliable external host.

    This is critical for systems that need to report data to external services
    or receive commands from remote systems. Network connectivity issues can
    cause data loss or prevent remote monitoring of the system.

    Returns:
        str or False: Returns ping response code as string if successful, False otherwise
    """
    hostname = "google.com"  # Reliable external host
    response = os.system("ping -c 1 -W 2 " + hostname)  # 2-second timeout
    if response == 0:
        return str(response)
    else:
        return False


def _healthcheck_temp_sensors():
    """
    Verifies that temperature sensors are operational and providing readings
    within expected ranges.

    Temperature sensors are critical for monitoring system health. Abnormal readings
    or sensor failures could indicate potential hardware issues or environmental problems
    that require immediate attention to prevent damage to equipment.

    Returns:
        dict: Status of upper and lower temperature sensors with readings
        False: If either sensor fails to provide valid readings
    """
    try:
        # Check upper temperature sensor
        temp_sensor_upper = TempSensor(spi_port=1, chip_select=1)
        temp_upper = temp_sensor_upper.read_temperature()
        temp_sensor_upper.close()

        # Check lower temperature sensor
        temp_sensor_lower = TempSensor(spi_port=1, chip_select=0)
        temp_lower = temp_sensor_lower.read_temperature()
        temp_sensor_lower.close()

        # Verify readings are within expected range: typically -20 to 125°C for most sensors
        # These are typical operating ranges - adjust based on your specific environment
        if (
            not math.isnan(temp_upper)
            and -20 <= temp_upper <= 125
            and not math.isnan(temp_lower)
            and -20 <= temp_lower <= 125
        ):
            return {
                "temperature_upper": temp_upper,
                "temperature_lower": temp_lower,
                "status": "OK",
            }
        else:
            logger.warning(
                f"Temperature sensor readings out of range: upper={temp_upper}, lower={temp_lower}"
            )
            return False
    except Exception as e:
        logger.error(f"Temperature sensor healthcheck failed: {str(e)}")
        return False


def _healthcheck_rpm_sensor():
    """
    Verifies that the RPM sensor is operational and providing plausible readings.

    The RPM sensor monitors critical rotating components. Failures in this sensor
    could prevent early detection of mechanical issues, potentially leading to
    catastrophic failures if components are operating outside of specification.

    Returns:
        dict: Current RPM reading and sensor status
        False: If the sensor fails to provide valid readings
    """
    try:
        rpm = rpm_sensor.read_rpm()

        # Check if RPM is a number and within a reasonable range
        # The exact range depends on your application - adjust as needed
        # For example, 0-10000 RPM might be reasonable for many fans/motors
        if isinstance(rpm, (int, float)) and 0 <= rpm <= 10000:
            return {"rpm": rpm, "status": "OK"}
        else:
            logger.warning(f"RPM sensor reading out of range: {rpm}")
            return False
    except Exception as e:
        logger.error(f"RPM sensor healthcheck failed: {str(e)}")
        return False


def _healthcheck_audio_sensor():
    """
    Verifies that the audio capture and processing system is operational.

    Audio analysis is used for acoustic monitoring which can detect abnormal
    operating conditions through sound pattern changes. A failed audio system
    could miss early indicators of mechanical wear or other issues that have
    distinctive acoustic signatures.

    Returns:
        dict: Status of audio processing components
        False: If audio capture or processing is not functioning correctly
    """
    try:
        # Check if we can get MFCC data (acoustic features)
        mfcc_data = audio_sensor.read_mfcc()

        # Check if we can get spectrum data
        spectrum_data = audio_sensor.read_spectrum()

        # Verify we have data in both
        if (
            mfcc_data
            and spectrum_data
            and isinstance(mfcc_data, dict)
            and isinstance(spectrum_data, dict)
            and len(mfcc_data) > 0
            and len(spectrum_data) > 0
        ):
            return {
                "audio_processing": "OK",
                "mfcc_features": len(mfcc_data),
                "spectrum_bands": len(spectrum_data),
            }
        else:
            logger.warning("Audio sensor not returning valid data")
            return False
    except Exception as e:
        logger.error(f"Audio sensor healthcheck failed: {str(e)}")
        return False


def _healthcheck_system_resources():
    """
    Monitors system resources to ensure adequate capacity for sensor operations.

    Resource constraints can cause missed readings, slow response times, or
    system instability. This check helps identify potential resource limitations
    before they affect system reliability.

    Returns:
        dict: Current system resource utilization metrics
        False: If resources are critically low
    """
    try:

        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.5)

        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent

        # Disk usage
        disk = psutil.disk_usage("/")
        disk_percent = disk.percent

        # Check for critical resource constraints
        # These thresholds should be adjusted based on your system requirements
        if cpu_percent < 90 and memory_percent < 90 and disk_percent < 95:
            return {
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent,
                "disk_percent": disk_percent,
                "status": "OK",
            }
        else:
            logger.warning(
                f"System resources critical: CPU={cpu_percent}%, Memory={memory_percent}%, Disk={disk_percent}%"
            )
            return False
    except ImportError:
        # psutil not installed, return minimal info
        logger.warning("psutil not installed, system resource check limited")
        return {"status": "UNKNOWN", "message": "psutil not installed"}
    except Exception as e:
        logger.error(f"System resource check failed: {str(e)}")
        return False


def _healthcheck_settings_integrity():
    """
    Verifies that sensor settings are within valid ranges and consistent.

    Configuration integrity is essential for proper sensor operation. Invalid
    settings can cause erroneous readings, sensor malfunction, or system
    instability. This check helps identify configuration issues before they
    cause operational problems.

    Returns:
        dict: Status of sensor settings
        False: If settings are invalid or inconsistent
    """
    try:
        # Check RPM sensor settings
        rpm_valid = (
            rpm_settings["measurement_window"] > 0
            and rpm_settings["measurement_interval"] > 0
            and rpm_settings["sample_size"] > 0
        )

        # Check audio settings
        audio_valid = (
            audio_settings["sample_duration"] > 0
            and audio_settings["mfcc_count"] > 0
            and audio_settings["buffer_size"] > 0
        )

        if rpm_valid and audio_valid:
            return {"rpm_settings": "valid", "audio_settings": "valid", "status": "OK"}
        else:
            issues = []
            if not rpm_valid:
                issues.append("RPM settings invalid")
            if not audio_valid:
                issues.append("Audio settings invalid")
            logger.warning(f"Settings integrity check failed: {', '.join(issues)}")
            return False
    except Exception as e:
        logger.error(f"Settings integrity check failed: {str(e)}")
        return False


app.add_api_route(
    "/health",
    health(
        [
            _healthcheck_ping,
            _healthcheck_temp_sensors,
            _healthcheck_rpm_sensor,
            _healthcheck_audio_sensor,
            _healthcheck_system_resources,
            _healthcheck_settings_integrity,
        ]
    ),
    summary="Perform comprehensive system health verification and hardware connectivity tests",
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

# Individual component health checks
app.add_api_route(
    "/health/network",
    health([_healthcheck_ping]),
    summary="Check network connectivity status",
    description="Verifies external network connectivity by pinging a reliable external host.",
    response_description="Returns HTTP 200 if network connectivity is available.",
)

app.add_api_route(
    "/health/temperature",
    health([_healthcheck_temp_sensors]),
    summary="Check temperature sensor status",
    description="Verifies that temperature sensors are operational and providing valid readings.",
    response_description="Returns HTTP 200 if temperature sensors are functioning correctly.",
)

app.add_api_route(
    "/health/rpm",
    health([_healthcheck_rpm_sensor]),
    summary="Check RPM sensor status",
    description="Verifies that the RPM sensor is operational and providing plausible readings.",
    response_description="Returns HTTP 200 if the RPM sensor is functioning correctly.",
)

app.add_api_route(
    "/health/audio",
    health([_healthcheck_audio_sensor]),
    summary="Check audio processing system status",
    description="Verifies that the audio capture and processing system is operational.",
    response_description="Returns HTTP 200 if audio sensors and processing are functioning correctly.",
)

app.add_api_route(
    "/health/system",
    health([_healthcheck_system_resources]),
    summary="Check system resource status",
    description="Monitors system resources to ensure adequate capacity for sensor operations.",
    response_description="Returns HTTP 200 if system resources are at acceptable levels.",
)
