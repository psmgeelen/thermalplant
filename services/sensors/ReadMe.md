# Sensors Service

A comprehensive Python-based sensors service that provides a unified API for various hardware sensor types. This service manages multiple sensor types and exposes their data through RESTful endpoints.

## Overview

The Sensors Service integrates with various hardware sensors and provides a unified API for reading and configuring them. It is designed to be robust, configurable, and extensible to support multiple sensor types:

- Temperature sensors (via SPI)
- RPM sensors (via GPIO)
- Voltage sensors (via I2C/ADS1115)
- Audio processing (for sound analysis)

## Features

- **RESTful API**: All sensor data is accessible through HTTP endpoints
- **Real-time monitoring**: Continuous sensor data sampling and processing
- **Configurable settings**: Runtime-adjustable sensor parameters
- **Health checks**: Built-in diagnostics for all sensor subsystems
- **Thread-safe logging**: Queue-based logging system for reliable operation
- **Rate limiting**: Prevents API abuse and ensures stable operation

## Sensor Types

### Temperature Sensors

The system supports temperature sensors connected via SPI:

- Configurable SPI port, chip select, and speed
- Temperature readings in Celsius
- Error handling for SPI communication issues
- Multiple sensor support (upper and lower temperature points)

### RPM Sensor

For measuring rotational speed:

- GPIO-based pulse detection
- Configurable measurement window and intervals
- Real-time RPM calculation with low-speed detection
- Thread-safe operation with continuous monitoring

### Voltage Sensor

For monitoring voltage levels:

- ADS1115 ADC-based voltage measurement
- Auto-adjusting gain for optimal precision
- I2C communication with configurable address
- Filtered measurements with averaging

### Audio Processing

For sound analysis:

- MFCC (Mel-frequency cepstral coefficients) calculation
- Spectrum analysis
- Configurable sampling rates and buffer sizes
- Hardware integration for audio capture

## API Endpoints

The service provides the following API endpoints:

- `/ping` - Health check endpoint
- `/sensors` - Consolidated readings from all available sensors
- `/temperature/upper` - Upper temperature sensor reading
- `/temperature/lower` - Lower temperature sensor reading
- `/rpm` - Current RPM reading
- `/rpm/settings` - Get/update RPM sensor settings
- `/voltage` - Current voltage reading
- `/voltage/settings` - Get/update voltage sensor settings
- `/audio/mfcc` - Audio MFCC data
- `/audio/spectrum` - Audio spectrum data
- `/audio/all` - All audio features
- `/audio/settings` - Get/update audio processing settings
- `/settings` - Get all sensor settings

## Configuration

Each sensor type has configurable settings that can be adjusted at runtime:

### RPM Sensor Settings

- `measurement_window`: Window size for measurements (in samples)
- `measurement_interval`: Interval between measurements (in seconds)
- `sample_size`: Number of samples needed to ensure not skipping a cycle

### Voltage Sensor Settings

- `i2c_address`: I2C address of the ADS1115 (default 0x48)
- `adc_channel`: ADC channel to read from (0-3)
- `measurement_window`: Window size for measurements (in samples)
- `measurement_interval`: Interval between measurements (in seconds)
- `sample_size`: Number of samples needed for reliable measurement

### Audio Settings

- `sample_duration`: Duration of audio samples for processing
- `mfcc_count`: Number of MFCC coefficients to extract
- `buffer_size`: Size of the audio processing buffer
- `n_bands`: Number of frequency bands for spectrum analysis

## Health Checks

The service includes comprehensive health checks for all sensor subsystems:

- Connection tests
- Reading validation
- Settings integrity checks
- System resource monitoring

## Logging

A thread-safe, queue-based logging system ensures that all events are properly recorded without affecting sensor performance:

- Configurable log levels
- Console and file output options
- Graceful shutdown handling

## Dependencies

- FastAPI: Web framework for API endpoints
- SpiDev: SPI communication for temperature sensors
- GPIOZero: GPIO access for RPM sensors
- Adafruit_ADS1x15: I2C communication for voltage sensors
- PyAudio/NumPy: Audio processing capabilities
- Pydantic: Data validation and settings management

## Installation and Setup

1. Install required dependencies
2. Configure hardware connections according to your setup
3. Adjust sensor settings as needed
4. Run the application using `uvicorn app.main:app`

## Hardware Requirements

- Raspberry Pi or compatible single-board computer
- SPI-connected temperature sensors
- GPIO-connected RPM sensor (hall effect or similar)
- I2C-connected ADS1115 for voltage measurement
- Audio input device (microphone or line-in)

## Development and Testing

The project includes unit tests for each sensor type to ensure proper functionality. Run tests using pytest:
