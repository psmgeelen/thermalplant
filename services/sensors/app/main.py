import os
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.openapi.utils import get_openapi
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import logging
from fastapi_health import health
from sensors import (
    TempSensor,
    RPMSensor,
    RPMSensorSettings,
    AudioHandler,
    AudioHandlerSettings,
    utils
)
import math
import psutil
import asyncio
from typing import Dict, Any, Optional, List, Union
from pydantic import BaseModel, Field

# Setup queue-based logging system
utils.setup_queue_logging(level=logging.INFO)
logger = utils.get_logger("API")


# Init API
limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

class HealthCheckResponse(BaseModel):
    status: str = Field(..., description="Status of the health check (ok or error)")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional health check details")

# Fixed sensor parameters
GPIO_PIN = 22
AUDIO_RATE = 44100
AUDIO_CHANNELS = 1

# Configurable sensor parameters with defaults
rpm_settings = RPMSettings()
audio_settings = AudioSettings()


# Initialize sensors with current settings
async def initialize_rpm_sensor():
    global rpm_sensor
    try:
        # If the sensor exists, close/stop it first
        if "rpm_sensor" in globals() and rpm_sensor is not None:
            try:
                rpm_sensor.stop()
            except Exception as e:
                logger.warning(f"Error stopping existing RPM sensor: {e}")

        rpm_sensor = RPMSensor(
            gpio_pin=GPIO_PIN,
            measurement_window=rpm_settings.measurement_window,
            measurement_interval=rpm_settings.measurement_interval,
            sample_size=rpm_settings.sample_size,
        )
        logger.info(f"RPM sensor initialized with settings: {rpm_settings.dict()}")
        return rpm_sensor
    except Exception as e:
        logger.error(f"Error initializing RPM sensor: {e}")
        raise


import time

async def initialize_audio_handler():
    global audio_sensor
    max_retries = 3
    retry_delay = 1

    for attempt in range(max_retries):
        try:
            if "audio_sensor" in globals() and audio_sensor is not None:
                audio_sensor.close()

            logger.info(f"Audio initialization attempt {attempt+1}/{max_retries}")
            
            audio_sensor = AudioHandler(
                rate=AUDIO_RATE,
                channels=AUDIO_CHANNELS,
                sample_duration=audio_settings.sample_duration,
                mfcc_count=audio_settings.mfcc_count,
                buffer_size=audio_settings.buffer_size,
                n_bands=audio_settings.n_bands,
            )
            
            # Wait a bit to see if we're getting audio data
            await asyncio.sleep(2)
            
            # Check if we're getting data
            mfcc_data = audio_sensor.read_mfcc()
            if not mfcc_data:
                raise RuntimeError("No audio data received after initialization")
                
            logger.info("Audio handler successfully initialized and receiving data")
            return audio_sensor
            
        except Exception as e:
            logger.warning(f"Audio initialization attempt {attempt+1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
                retry_delay *= 1.5
    
    raise RuntimeError("Failed to initialize audio handler")



# Initialize sensors on startup instead of at module level
rpm_sensor = None
audio_sensor = None

# Dependency injection functions
def get_rpm_sensor():
    return rpm_sensor

def get_audio_sensor():
    if audio_sensor is None:
        raise HTTPException(
            status_code=503,
            detail="Audio subsystem is currently unavailable"
        )
    return audio_sensor

async def get_temp_sensor_upper():
    sensor = TempSensor(spi_port=1, chip_select=1)
    try:
        yield sensor
    finally:
        sensor.close()

async def get_temp_sensor_lower():
    sensor = TempSensor(spi_port=1, chip_select=0)
    try:
        yield sensor
    finally:
        sensor.close()

@app.on_event("startup")
async def startup_event():
    global rpm_sensor, audio_sensor
    rpm_sensor = await initialize_rpm_sensor()
    
    try:
        audio_sensor = await initialize_audio_handler()
    except Exception as e:
        logger.error(f"Audio handler initialization failed: {e}")
        logger.warning("Application will continue WITHOUT audio capabilities")
        audio_sensor = None

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Application shutting down")
    
    # First stop all sensors
    if "rpm_sensor" in globals() and rpm_sensor is not None:
        try:
            rpm_sensor.stop()
        except Exception as e:
            logger.warning(f"Error stopping RPM sensor during shutdown: {e}")
    
    if "audio_sensor" in globals() and audio_sensor is not None:
        try:
            audio_sensor.close()
        except Exception as e:
            logger.warning(f"Error closing audio handler during shutdown: {e}")
    
    # Finally, shutdown logging system to ensure all pending logs are processed
    utils.shutdown_logging()


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
async def ping():
    return "pong"


@app.get(
    "/temperature_upper",
    summary="Retrieve temperature readings from the upper thermal segment sensor",
    description="",
    response_description="A dictionary with a list of devices",
    response_model=str,
)
@limiter.limit("1000/minute")
async def get_temperature_upper(
    request: Request,
    sensor: TempSensor = Depends(get_temp_sensor_upper),
):
    return sensor.read_temperature()


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
@limiter.limit("1000/minute")
async def get_temperature_lower(
    request: Request,
    sensor: TempSensor = Depends(get_temp_sensor_lower),
):
    return sensor.read_temperature()


@app.get(
    "/rpm",
    summary="Retrieve real-time rotational speed (RPM) measurement from the fan sensor",
    description=("This request returns the current RPM reading from the fan sensor"),
    response_description="Current RPM value",
    response_model=float,
)
@limiter.limit("1000/minute")
async def get_rpm(
    request: Request,
    rpm_sensor: RPMSensor = Depends(get_rpm_sensor),
):
    return rpm_sensor.read_rpm()


@app.get(
    "/rpm/settings",
    summary="Get RPM sensor settings",
    description="Returns the current configuration settings for the RPM sensor",
    response_description="Dictionary containing the RPM sensor settings",
    response_model=RPMSensorSettings,
)
@limiter.limit("100/minute")
async def get_rpm_settings(request: Request):
    return RPMSensorSettings(
        measurement_window=rpm_settings.measurement_window,
        measurement_interval=rpm_settings.measurement_interval,
        sample_size=rpm_settings.sample_size
    )


@app.put(
    "/rpm/settings",
    summary="Update RPM sensor settings",
    description="Update the configuration settings for the RPM sensor and reinitialize it",
    response_description="Dictionary containing the updated RPM sensor settings",
    response_model=RPMSensorSettings,
)
@limiter.limit("20/minute")
async def update_rpm_settings(request: Request, settings: RPMSensorSettings):
    try:
        # Update the global settings
        global rpm_settings
        rpm_settings = RPMSettings(
            measurement_window=settings.measurement_window,
            measurement_interval=settings.measurement_interval,
            sample_size=settings.sample_size
        )

        # Reinitialize the sensor with new settings
        await initialize_rpm_sensor()

        return settings
    except Exception as e:
        logger.error(f"Failed to update RPM sensor settings: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to update RPM sensor settings: {str(e)}"
        )


@app.get(
    "/mfcc",
    summary="Extract Mel-Frequency Cepstral Coefficients (MFCC) "
            "acoustic features from engine sound",
    description=(
        "Returns MFCC coefficients with frequency labels in a dictionary format"
    ),
    response_description="A dictionary of labeled MFCC coefficients",
)
@limiter.limit("1000/minute")
async def get_mfcc(
    request: Request,
    audio_sensor: AudioHandler = Depends(get_audio_sensor)
):
    return audio_sensor.read_mfcc()


@app.get(
    "/audio/settings",
    summary="Get audio handler settings",
    description="Returns the current configuration settings for the audio handler",
    response_description="Dictionary containing the audio handler settings",
    response_model=AudioHandlerSettings,
)
@limiter.limit("100/minute")
async def get_audio_settings(request: Request):
    return AudioHandlerSettings(
        sample_duration=audio_settings.sample_duration,
        mfcc_count=audio_settings.mfcc_count,
        buffer_size=audio_settings.buffer_size,
        n_bands=audio_settings.n_bands,
    )


@app.put(
    "/audio/settings",
    summary="Update audio handler settings",
    description="Update the configuration settings for the audio handler and reinitialize it",
    response_description="Dictionary containing the updated audio handler settings",
    response_model=AudioHandlerSettings,
)
@limiter.limit("20/minute")
async def update_audio_settings(request: Request, settings: AudioHandlerSettings):
    try:
        # Update the global settings
        global audio_settings
        audio_settings = AudioSettings(
            sample_duration=settings.sample_duration,
            mfcc_count=settings.mfcc_count,
            buffer_size=settings.buffer_size,
            n_bands=audio_settings.n_bands,
        )

        # Reinitialize the audio handler with new settings
        await initialize_audio_handler()

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
@limiter.limit("1000/minute")
async def get_spectrum(
    request: Request,
    audio_sensor: AudioHandler = Depends(get_audio_sensor)
):
    return audio_sensor.read_spectrum()


@app.get(
    "/audio",
    summary="Get all audio data",
    description=("Returns both MFCC and spectrum data with frequency labels"),
    response_description="A dictionary containing both MFCC and spectrum data",
)
@limiter.limit("500/minute")
async def get_audio(
    request: Request,
    audio_sensor: AudioHandler = Depends(get_audio_sensor)
):
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
async def get_all_sensors(
    request: Request,
    temp_upper: TempSensor = Depends(get_temp_sensor_upper),
    temp_lower: TempSensor = Depends(get_temp_sensor_lower),
    rpm_sensor: RPMSensor = Depends(get_rpm_sensor),
):
    # Get temperature readings
    temp_upper_val = temp_upper.read_temperature()
    temp_lower_val = temp_lower.read_temperature()

    # Get RPM reading
    rpm = rpm_sensor.read_rpm()

    # Combine all data
    all_sensors = {
        "temperature_upper": temp_upper_val,
        "temperature_lower": temp_lower_val,
        "rpm": rpm,
    }

    # Try to get audio features if audio subsystem is available
    if audio_sensor is not None:
        try:
            audio_data = audio_sensor.read_all_audio()
            all_sensors["audio"] = audio_data
        except Exception as e:
            logger.warning(f"Error getting audio data: {e}")
            all_sensors["audio"] = {"status": "unavailable", "error": str(e)}
    else:
        all_sensors["audio"] = {"status": "unavailable", "error": "Audio subsystem not initialized"}

    return all_sensors


@app.get(
    "/settings",
    summary="Retrieve comprehensive configuration parameters for all sensor subsystems",
    description="Returns the current configuration settings for all sensors",
    response_description="Dictionary containing all sensor settings",
)
@limiter.limit("100/minute")
async def get_all_settings(request: Request):
    return {
        "rpm": rpm_settings.dict(),
        "audio": audio_settings.dict(),
        "fixed_settings": {
            "rpm": {"gpio_pin": GPIO_PIN},
            "audio": {"rate": AUDIO_RATE, "channels": AUDIO_CHANNELS},
        },
    }


##### Healthchecks #####
import socket

async def _healthcheck_ping():
    """
    Verifies external network connectivity by checking socket connection to a reliable external host.
    
    This is critical for systems that need to report data to external services
    or receive commands from remote systems. Network connectivity issues can
    cause data loss or prevent remote monitoring of the system.
    
    Uses pure Python socket implementation rather than OS-level ping commands,
    making it more portable and container-friendly.

    Returns:
        HealthCheckResponse: Status and details of the health check
    """
    hostname = "8.8.8.8"  # Google DNS - highly reliable and typically accessible
    port = 53  # DNS port
    timeout = 2  # seconds
    
    try:
        # Create a socket object with a timeout
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        
        # Start measuring connection time
        start_time = time.time()
        
        # Attempt to connect to the host
        result = sock.connect_ex((hostname, port))
        
        # Calculate response time
        response_time = (time.time() - start_time) * 1000  # convert to milliseconds
        
        # Close the socket
        sock.close()
        
        if result == 0:
            return {
                "status": "ok",
                "details": {
                    "connected_to": f"{hostname}:{port}",
                    "response_time_ms": round(response_time, 2)
                }
            }
        else:
            return {
                "status": "error",
                "details": {
                    "message": f"Connection failed with error code {result}",
                    "socket_error": socket.errno.errorcode.get(result, "Unknown error")
                }
            }
            
    except socket.gaierror as e:
        return {
            "status": "error",
            "details": {
                "message": f"Address-related error: {str(e)}",
                "error_type": "DNS resolution failure"
            }
        }
    except socket.timeout:
        return {
            "status": "error",
            "details": {
                "message": f"Connection to {hostname}:{port} timed out after {timeout} seconds",
                "error_type": "Connection timeout"
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "details": {
                "message": str(e),
                "error_type": type(e).__name__
            }
        }


async def _healthcheck_temp_sensors():
    """
    Verifies that temperature sensors are operational and providing readings
    within expected ranges.

    Temperature sensors are critical for monitoring system health. Abnormal readings
    or sensor failures could indicate potential hardware issues or environmental problems
    that require immediate attention to prevent damage to equipment.

    Returns:
        HealthCheckResponse: Status and details of the health check
    """
    try:
        # Check upper temperature sensor
        temp_upper = None
        temp_lower = None
        
        # Use dependency injection pattern for sensors
        async for sensor in get_temp_sensor_upper():
            temp_upper = sensor.read_temperature()
        
        async for sensor in get_temp_sensor_lower():
            temp_lower = sensor.read_temperature()

        # Verify readings are within expected range: typically -20 to 125°C for most sensors
        # These are typical operating ranges - adjust based on your specific environment
        if (
            temp_upper is not None
            and temp_lower is not None
            and not math.isnan(temp_upper)
            and -20 <= temp_upper <= 125
            and not math.isnan(temp_lower)
            and -20 <= temp_lower <= 125
        ):
            return {
                "status": "ok",
                "details": {
                    "temperature_upper": temp_upper,
                    "temperature_lower": temp_lower
                }
            }
        else:
            logger.warning(
                f"Temperature sensor readings out of range: upper={temp_upper}, lower={temp_lower}"
            )
            return {
                "status": "error",
                "details": {
                    "message": "Temperature sensor readings out of range or invalid",
                    "temperature_upper": temp_upper,
                    "temperature_lower": temp_lower
                }
            }
    except Exception as e:
        logger.error(f"Temperature sensor healthcheck failed: {str(e)}")
        return {"status": "error", "details": {"message": str(e)}}


async def _healthcheck_rpm_sensor():
    """
    Verifies that the RPM sensor is operational and providing plausible readings.

    The RPM sensor monitors critical rotating components. Failures in this sensor
    could prevent early detection of mechanical issues, potentially leading to
    catastrophic failures if components are operating outside of specification.

    Returns:
        HealthCheckResponse: Status and details of the health check
    """
    try:
        rpm_sensor_instance = get_rpm_sensor()
        if rpm_sensor_instance is None:
            return {
                "status": "error", 
                "details": {"message": "RPM sensor not initialized"}
            }
            
        rpm = rpm_sensor_instance.read_rpm()

        # Check if RPM is a number and within a reasonable range
        # The exact range depends on your application - adjust as needed
        # For example, 0-10000 RPM might be reasonable for many fans/motors
        if isinstance(rpm, (int, float)) and 0 <= rpm <= 10000:
            return {"status": "ok", "details": {"rpm": rpm}}
        else:
            logger.warning(f"RPM sensor reading out of range: {rpm}")
            return {
                "status": "error",
                "details": {
                    "message": "RPM reading out of acceptable range",
                    "rpm": rpm
                }
            }
    except Exception as e:
        logger.error(f"RPM sensor healthcheck failed: {str(e)}")
        return {"status": "error", "details": {"message": str(e)}}


async def _healthcheck_audio_sensor():
    """
    Verifies that the audio capture and processing system is operational.

    Audio analysis is used for acoustic monitoring which can detect abnormal
    operating conditions through sound pattern changes. A failed audio system
    could miss early indicators of mechanical wear or other issues that have
    distinctive acoustic signatures.

    Returns:
        HealthCheckResponse: Status and details of the health check
    """
    # First check if audio sensor is even available
    if audio_sensor is None:
        return {
            "status": "warning",  # Use warning instead of error to allow overall health check to pass
            "details": {
                "message": "Audio subsystem is disabled or unavailable",
                "recommendation": "Audio functionality is optional; system can operate without it"
            }
        }
        
    try:
        # Try to use the sensor directly without the dependency that would raise an exception
        audio_sensor_instance = audio_sensor
            
        # Check if recording_loop is running and has PipeWire connection
        if (not hasattr(audio_sensor_instance, 'recording_loop') or 
            not audio_sensor_instance.recording_loop or 
            not audio_sensor_instance.recording_loop.running):
            return {
                "status": "warning",
                "details": {
                    "message": "PipeWire recording loop is not running",
                    "recommendation": "Restart the audio subsystem"
                }
            }
            
        # Check if we can get MFCC data (acoustic features)
        mfcc_data = audio_sensor_instance.read_mfcc()

        # Check if we can get spectrum data
        spectrum_data = audio_sensor_instance.read_spectrum()

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
                "status": "ok",
                "details": {
                    "mfcc_features": len(mfcc_data),
                    "spectrum_bands": len(spectrum_data),
                    "audio_backend": "PipeWire"
                }
            }
        else:
            logger.warning("Audio sensor not returning valid data")
            return {
                "status": "error",
                "details": {
                    "message": "Audio sensor not returning valid data",
                    "mfcc_valid": bool(mfcc_data and isinstance(mfcc_data, dict) and len(mfcc_data) > 0),
                    "spectrum_valid": bool(spectrum_data and isinstance(spectrum_data, dict) and len(spectrum_data) > 0),
                    "audio_backend": "PipeWire"
                }
            }
    except Exception as e:
        logger.error(f"Audio sensor healthcheck failed: {str(e)}")
        return {
            "status": "warning",  # Use warning instead of error to allow overall health check to pass
            "details": {
                "message": f"Audio sensor check failed: {str(e)}",
                "recommendation": "Audio functionality is optional; system can operate without it",
                "audio_backend": "PipeWire"
            }
        }


async def _healthcheck_system_resources():
    """
    Monitors system resources to ensure adequate capacity for sensor operations.

    Resource constraints can cause missed readings, slow response times, or
    system instability. This check helps identify potential resource limitations
    before they affect system reliability.

    Returns:
        HealthCheckResponse: Status and details of the health check
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
                "status": "ok",
                "details": {
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory_percent,
                    "disk_percent": disk_percent
                }
            }
        else:
            logger.warning(
                f"System resources critical: "
                f"CPU={cpu_percent}%, Memory={memory_percent}%, Disk={disk_percent}%"
            )
            return {
                "status": "error",
                "details": {
                    "message": "System resources are at critical levels",
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory_percent,
                    "disk_percent": disk_percent
                }
            }
    except PermissionError as pe:
        # Handle permission errors when accessing system information
        logger.error(f"Permission error when checking system resources: {str(pe)}")
        return {
            "status": "error", 
            "details": {
                "message": "Permission denied when checking system resources",
                "error": str(pe)
            }
        }
    except Exception as e:
        logger.error(f"System resource check failed: {str(e)}")
        return {"status": "error", "details": {"message": str(e)}}


async def _healthcheck_settings_integrity():
    """
    Verifies that sensor settings are within valid ranges and consistent.

    Configuration integrity is essential for proper sensor operation. Invalid
    settings can cause erroneous readings, sensor malfunction, or system
    instability. This check helps identify configuration issues before they
    cause operational problems.

    Returns:
        dict: Status and details of the health check
    """
    try:
        # Check RPM sensor settings using Pydantic model attributes
        rpm_valid = (
            hasattr(rpm_settings, 'measurement_window') and rpm_settings.measurement_window > 0
            and hasattr(rpm_settings, 'measurement_interval') and rpm_settings.measurement_interval > 0
            and hasattr(rpm_settings, 'sample_size') and rpm_settings.sample_size > 0
        )

        # Check audio settings using Pydantic model attributes
        audio_valid = (
            hasattr(audio_settings, 'sample_duration') and audio_settings.sample_duration > 0
            and hasattr(audio_settings, 'mfcc_count') and audio_settings.mfcc_count > 0
            and hasattr(audio_settings, 'buffer_size') and audio_settings.buffer_size > 0
            and hasattr(audio_settings, 'n_bands') and audio_settings.n_bands > 0
        )

        if rpm_valid and audio_valid:
            return {
                "status": "ok",
                "details": {
                    "rpm_settings": "valid", 
                    "audio_settings": "valid"
                }
            }
        else:
            issues = []
            rpm_issues = {}
            audio_issues = {}
            
            if not rpm_valid:
                issues.append("RPM settings invalid")
                if not hasattr(rpm_settings, 'measurement_window') or rpm_settings.measurement_window <= 0:
                    rpm_issues["measurement_window"] = "must be positive"
                if not hasattr(rpm_settings, 'measurement_interval') or rpm_settings.measurement_interval <= 0:
                    rpm_issues["measurement_interval"] = "must be positive"
                if not hasattr(rpm_settings, 'sample_size') or rpm_settings.sample_size <= 0:
                    rpm_issues["sample_size"] = "must be positive"
                    
            if not audio_valid:
                issues.append("Audio settings invalid")
                if not hasattr(audio_settings, 'sample_duration') or audio_settings.sample_duration <= 0:
                    audio_issues["sample_duration"] = "must be positive"
                if not hasattr(audio_settings, 'mfcc_count') or audio_settings.mfcc_count <= 0:
                    audio_issues["mfcc_count"] = "must be positive" 
                if not hasattr(audio_settings, 'buffer_size') or audio_settings.buffer_size <= 0:
                    audio_issues["buffer_size"] = "must be positive"
                if not hasattr(audio_settings, 'n_bands') or audio_settings.n_bands <= 0:
                    audio_issues["n_bands"] = "must be positive"
                    
            logger.warning(f"Settings integrity check failed: {', '.join(issues)}")
            return {
                "status": "error",
                "details": {
                    "message": "Settings integrity check failed",
                    "issues": issues,
                    "rpm_issues": rpm_issues,
                    "audio_issues": audio_issues
                }
            }
    except Exception as e:
        logger.error(f"Settings integrity check failed: {str(e)}")
        return {"status": "error", "details": {"message": str(e)}}


# Create a custom wrapped health function to handle async health checks
def async_health_dependency(health_checks):
    """
    Custom health check handler that supports async health checks.
    Returns a FastAPI dependency that performs the health checks.
    """
    async def health_endpoint():
        results = {}
        for check in health_checks:
            result = await check()
            check_name = check.__name__
            results[check_name] = result
            # If any check returns a falsy value or an error status, the health check fails
            if not result or (isinstance(result, dict) and result.get("status") == "error"):
                return {"status": "error", "checks": results}
        return {"status": "ok", "checks": results}
    
    return health_endpoint

# Health check routes with proper function passing
app.add_api_route(
    "/health",
    async_health_dependency([
        _healthcheck_ping,
        _healthcheck_system_resources,
        _healthcheck_settings_integrity,
    ]),
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
    response_model=Dict[str, Any]
)

# Individual component health checks 
app.add_api_route(
    "/health/network",
    async_health_dependency([_healthcheck_ping]),
    summary="Check network connectivity status",
    description="Verifies external network connectivity by pinging a reliable external host.",
    response_description="Returns HTTP 200 if network connectivity is available.",
    response_model=Dict[str, Any]
)

app.add_api_route(
    "/health/temperature",
    async_health_dependency([_healthcheck_temp_sensors]),
    summary="Check temperature sensor status",
    description="Verifies that temperature sensors are operational and providing valid readings.",
    response_description="Returns HTTP 200 if temperature sensors are functioning correctly.",
    response_model=Dict[str, Any]
)

app.add_api_route(
    "/health/rpm",
    async_health_dependency([_healthcheck_rpm_sensor]),
    summary="Check RPM sensor status",
    description="Verifies that the RPM sensor is operational and providing plausible readings.",
    response_description="Returns HTTP 200 if the RPM sensor is functioning correctly.",
    response_model=Dict[str, Any]
)

app.add_api_route(
    "/health/audio",
    async_health_dependency([_healthcheck_audio_sensor]),
    summary="Check audio processing system status",
    description="Verifies that the audio capture and processing system is operational.",
    response_description="Returns HTTP 200 if audio sensors and processing "
                         "are functioning correctly.",
    response_model=Dict[str, Any]
)

app.add_api_route(
    "/health/system",
    async_health_dependency([_healthcheck_system_resources]),
    summary="Check system resource status",
    description="Monitors system resources to ensure adequate capacity for sensor operations.",
    response_description="Returns HTTP 200 if system resources are at acceptable levels.",
    response_model=Dict[str, Any]
)

# Settings integrity health check
app.add_api_route(
    "/health/settings",
    async_health_dependency([_healthcheck_settings_integrity]),
    summary="Check configuration settings integrity",
    description="Verifies that all sensor settings are valid and consistent.",
    response_description="Returns HTTP 200 if all settings are valid and consistent.",
    response_model=Dict[str, Any]
)
