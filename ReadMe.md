# Thermal Plant Monitoring System

A comprehensive monitoring solution for thermal-based manufacturing plants, designed to showcase predictive maintenance capabilities using industrial OPC-UA architecture, time-series data storage, and advanced analytics.

## Project Overview

This project implements a complete sensor monitoring system for thermal plants, providing real-time data collection, storage, visualization, and predictive analytics. The system leverages industrial-standard OPC-UA protocol to connect various sensors and expose their data through a standardized interface. All collected data is stored in TimescaleDB, a specialized time-series database built on PostgreSQL, and visualized through Grafana dashboards.

## System Architecture

The system is composed of three main layers:

### 1. Data Collection Layer

- **Sensors Service**: Interfaces directly with hardware sensors (temperature, RPM, voltage, audio) and exposes their data through RESTful APIs
- **DMZ Nginx**: Provides a secure gateway to sensor data for the OPC UA server
- **OPC UA Server**: Implements an industrial-standard OPC UA server that transforms sensor data into a standardized format

### 2. Data Storage Layer

- **OPC Scraper**: Connects to the OPC UA server, retrieves sensor data, and stores it in TimescaleDB with intelligent traversal of OPC UA node hierarchies
- **TimescaleDB**: Specialized time-series database built on PostgreSQL for efficient storage and querying of sensor data with hypertable partitioning
- **Continuous Aggregations**: Automatically calculates rolling averages and other metrics at different time intervals (5-second, 1-minute, and custom periods)
- **Retention Policies**: Implements automatic data management to maintain performance while preserving historical data
- **Transaction Management**: Ensures data integrity with proper transaction handling and error recovery mechanisms

### 3. Visualization & Analytics Layer

- **Grafana**: Provides dashboards for visualizing sensor data and monitoring system health
- **Prometheus**: Collects infrastructure metrics for system performance monitoring
- **Node Exporter & cAdvisor**: Expose host and container metrics to Prometheus

## Repository Structure

```
/
├── ci/                  # Continuous Integration configurations
├── docs/                # Project documentation
├── nginx/               # Nginx configuration for the DMZ
├── prometheus/          # Prometheus configuration
├── services/            # Main services
│   ├── sensors/         # Hardware sensor interfaces and APIs
│   ├── opc_server/      # OPC UA server implementation
│   └── opc_scraper/     # Service to collect data from OPC UA and store in TimescaleDB
├── .env.template        # Template for environment variables
├── docker-compose.yaml  # Docker Compose configuration
├── LICENSE              # Project license
├── README.md            # This file
└── Taskfile.yaml        # Task runner configuration for automation
```

## Core Services

### Sensors Service

Connects to physical hardware sensors and provides a RESTful API for accessing sensor data. Supports:

- Temperature sensors (via SPI)
- RPM sensors (via GPIO)
- Voltage sensors (via I2C/ADS1115)
- Audio processing (for sound analysis)

Main endpoints include `/temperature/upper`, `/temperature/lower`, `/rpm`, `/voltage`, `/audio/mfcc`, `/audio/spectrum`, and more.

### OPC UA Server

Implements an OPC UA server that fetches data from the Sensors Service and exposes it through the standardized OPC UA protocol. Features:

- Standards-compliant OPC UA server implementation
- Multiple sensor type support
- Historical data storage
- Dynamic node structure creation
- Regular polling and updates

Clients can connect to the server at `opc.tcp://[server-ip]:4840/freeopcua/server/`.

#### OPC UA Node Structure

```
Server Root
├── Objects
│   ├── Server
│   └── Plant
│       ├── Temperature
│       │   ├── Upper (Variable)
│       │   └── Lower (Variable)
│       ├── RPM
│       │   └── MainShaft (Variable)
│       ├── Voltage
│       │   └── MainSupply (Variable)
│       └── Audio
│           ├── MFCC
│           │   ├── Coefficient_0 (Variable)
│           │   ├── Coefficient_1 (Variable)
│           │   └── ... (Additional coefficients)
│           └── Spectrum
│               ├── Band_0 (Variable)
│               ├── Band_1 (Variable)
│               └── ... (Additional bands)
└── Types
    └── DataTypes
        └── BaseDataType
            └── ... (Standard OPC UA types)
```

Each variable node includes the following attributes:
- **Value**: Current sensor reading
- **DataType**: Type of the value (Double, Int32, etc.)
- **AccessLevel**: Read/write permissions
- **Historizing**: Whether historical data is stored

The server also implements browsing capabilities allowing clients to discover the available nodes programmatically.

### OPC Scraper

Connects to the OPC UA server, retrieves sensor data, and stores it in a TimescaleDB database. Includes:

- OPC UA client integration
- Hierarchical node traversal
- Automatic database schema management
- Continuous data collection
- Error resilience and automatic recovery
- Data aggregation and resource management

## Monitoring Infrastructure

### TimescaleDB

Specialized time-series database built on PostgreSQL, configured with:

- Hypertables for efficient time-series data storage
- Continuous aggregations for time-based data rollups
- Retention policies to prevent unbounded data growth

### Prometheus

Monitors system metrics with a 1-second scrape interval. Configured to collect metrics from:

- Node Exporter (host system metrics)
- cAdvisor (container metrics)
- PostgreSQL Exporter (database metrics)

### Grafana

Provides dashboards for visualizing sensor data and system metrics. Includes:

- Time-series graphs for temperature, pressure, and flow rate
- System performance monitoring
- Alert configuration

## Hardware Requirements

- Raspberry Pi 5 (or compatible single-board computer)
- Temperature sensors connected via SPI
- RPM sensor connected via GPIO
- Voltage sensors connected via I2C (ADS1115)
- Audio input device (microphone or line-in)

### Hardware Integration Details

```
+---------------------+   +-----------------+   +--------------------+
| Raspberry Pi 5      |   | Connection Type |   | Sensor Type        |
+---------------------+   +-----------------+   +--------------------+
|                     |   |                 |   |                    |
| GPIO 23 ------------|-->| Digital Input   |-->| Hall Effect Sensor |
|                     |   |                 |   | (RPM Measurement)  |
|                     |   |                 |   |                    |
| SPI0 CE0 -----------|-->| SPI Interface   |-->| MAX31855          |
| (GPIO 8)            |   |                 |   | (Upper Temperature)|
|                     |   |                 |   |                    |
| SPI0 CE1 -----------|-->| SPI Interface   |-->| MAX31855          |
| (GPIO 7)            |   |                 |   | (Lower Temperature)|
|                     |   |                 |   |                    |
| I2C1 SDA/SCL -------|-->| I2C Interface   |-->| ADS1115 ADC       |
| (GPIO 2/3)          |   |                 |   | (Voltage Sensing)  |
|                     |   |                 |   |                    |
| USB Port -----------|-->| Audio Interface |-->| USB Microphone     |
|                     |   |                 |   | (Audio Analysis)   |
+---------------------+   +-----------------+   +--------------------+
```

#### Temperature Sensor Setup

The MAX31855 thermocouples are connected to the Raspberry Pi via SPI interface with the following pinout:

- **Upper Sensor**: SPI0 CE0 (GPIO 8), supporting temperatures from -200°C to +1350°C
- **Lower Sensor**: SPI0 CE1 (GPIO 7), supporting temperatures from -200°C to +1350°C

#### RPM Sensor Configuration

The Hall Effect sensor is connected to GPIO 23 and configured to detect magnetic pulses as the shaft rotates:

- Sensor outputs a digital pulse with each rotation
- GPIO pin is configured with a pull-up resistor and interrupt handler
- RPM calculation: `RPM = (pulses_counted / pulse_per_revolution) * (60 / measurement_time_seconds)`

#### Voltage Monitoring

Voltage is measured using an ADS1115 16-bit ADC connected via I2C:

- **I2C Address**: 0x48 (default)
- **Resolution**: 16 bits (0.125mV per bit at ±4.096V range)
- **Sampling Rate**: 860 samples per second
- **Voltage Divider**: External voltage divider for measuring higher voltages

## Getting Started

### Prerequisites

- Docker and Docker Compose
- Raspberry Pi with GPIO access
- Connected sensors (or simulated sensors for testing)

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/thermal-plant-monitoring.git
   cd thermal-plant-monitoring
   ```

2. Set up environment variables:
   ```bash
   cp .env.template .env
   # Edit .env with your configuration
   ```

3. Install system dependencies and tools (using Task):
   ```bash
   task install_docker
   task install_pyenv
   task install_poetry
   ```

4. Deploy the full stack:
   ```bash
   task deploy_full_stack
   ```

### Deployment Profiles

The system supports different deployment profiles via Docker Compose:

- `fullstack`: Deploys all services
- `infrastructure`: Deploys only monitoring infrastructure (Grafana, Prometheus, etc.)
- `opc`: Deploys only OPC-related services
- `sensors_only`: Deploys only the sensors service for testing

Example:
```bash
docker compose --profile=infrastructure up -d
```

## Data Visualization

Access Grafana at http://localhost:3000 to view dashboards. Default credentials are admin/admin.

1. Create a new data source for Prometheus (http://localhost:9090)
2. Create a new data source for TimescaleDB
3. Import or create dashboards for visualizing sensor data

### Example Dashboard Layouts

```
+-----------------------------------+-----------------------------------+
|                                   |                                   |
|   Temperature Trends (Line Chart) |   RPM vs Temperature (Scatter)    |
|                                   |                                   |
+-----------------------------------+-----------------------------------+
|                                   |                                   |
|   System Health (Gauges)          |   Anomaly Detection (Heatmap)     |
|                                   |                                   |
+-----------------------------------+-----------------------------------+
|                                                                       |
|                                                                       |
|   Audio Spectrum Analysis (Spectrogram)                               |
|                                                                       |
+-----------------------------------------------------------------------+
|                                                                       |
|   Alert History and Status (Table)                                    |
|                                                                       |
+-----------------------------------------------------------------------+
```

### Example SQL Queries for Dashboards

#### Recent Temperature Readings
```sql
SELECT
  time_bucket('5s', timestamp) AS time,
  avg(value) AS temperature
FROM sensor_data
WHERE
  sensorname = 'temperature' AND
  tagname = 'upper' AND
  timestamp > now() - interval '1 hour'
GROUP BY time
ORDER BY time;
```

#### Anomaly Detection
```sql
SELECT
  time_bucket('1m', timestamp) AS time,
  avg(value) AS avg_value,
  stddev(value) AS stddev_value
FROM sensor_data
WHERE
  sensorname = 'rpm' AND
  timestamp > now() - interval '24 hours'
GROUP BY time
ORDER BY time;
```

#### Multiple Sensor Correlation
```sql
WITH temp_data AS (
  SELECT
    time_bucket('1m', timestamp) AS time,
    avg(value) AS temperature
  FROM sensor_data
  WHERE
    sensorname = 'temperature' AND
    tagname = 'upper' AND
    timestamp > now() - interval '6 hours'
  GROUP BY time
),
rpm_data AS (
  SELECT
    time_bucket('1m', timestamp) AS time,
    avg(value) AS rpm
  FROM sensor_data
  WHERE
    sensorname = 'rpm' AND
    timestamp > now() - interval '6 hours'
  GROUP BY time
)
SELECT
  t.time,
  t.temperature,
  r.rpm
FROM temp_data t
JOIN rpm_data r ON t.time = r.time
ORDER BY t.time;
```

## Continuous Integration

The project includes CI configuration for automated testing and deployment. Run:

```bash
task ci
```

## Database Management

TimescaleDB includes continuous aggregates for efficient time-series analysis:

```sql
-- Create TimescaleDB toolkit extension for advanced features
CREATE EXTENSION IF NOT EXISTS timescaledb_toolkit;
```

## Resources

- OPC-UA Library: 
  - Repo: https://github.com/FreeOpcUa/opcua-asyncio
  - Documentation: https://opcua-asyncio.readthedocs.io/en/latest/
- TimescaleDB Documentation: https://docs.timescale.com/
- Grafana Documentation: https://grafana.com/docs/

## Contributing

We welcome contributions to the Thermal Plant Monitoring System! Here's how you can help:

```
+------------------------+  +-----------------------+  +-----------------------+
|                        |  |                       |  |                       |
|   Report Issues        |  |   Submit PRs          |  |   Suggest Features    |
|                        |  |                       |  |                       |
| - Bug reports          |  | - Bug fixes           |  | - New sensor types    |
| - Performance issues   |  | - Feature additions   |  | - Dashboard ideas     |
| - Documentation gaps   |  | - Documentation       |  | - Integration options |
|                        |  | - Tests               |  | - Analytics           |
+------------------------+  +-----------------------+  +-----------------------+
```

### Development Workflow

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Coding Standards

- Follow PEP 8 style guide for Python code
- Write docstrings for all functions, classes, and modules
- Add appropriate tests for new functionality
- Ensure all Docker containers are properly configured
- Update documentation to reflect changes

### Testing

Before submitting PRs, please ensure:

1. All tests pass
2. New features include appropriate tests
3. The system deploys successfully with your changes
4. No regressions are introduced

## License

This project is licensed under the MIT License - see the LICENSE file for details.