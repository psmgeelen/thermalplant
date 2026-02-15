# OPC Server Service

An OPC UA server implementation that exposes sensor data from multiple sources through a standardized industrial protocol.

## Overview

The OPC Server service acts as a bridge between sensor data APIs and industrial control systems by implementing an OPC UA server. It fetches data from various sensor endpoints, makes them available through OPC UA, and provides historical data storage capabilities.

## Features

- **OPC UA Server**: Implements a standards-compliant OPC UA server
- **Multiple Sensor Support**: Integrates various sensor types (temperature, RPM, voltage, audio)
- **Historical Data**: Stores sensor readings in SQLite for historical analysis
- **Dynamic Node Structure**: Creates OPC UA nodes based on available sensors
- **Regular Polling**: Continuously fetches and updates sensor values

## Architecture

The service has a modular architecture consisting of:

1. **OPC UA Server**: Core server implementation using `asyncua`
2. **History Storage**: SQLite-based historical data storage
3. **Sensor Adapters**: Classes that fetch and format data from sensor APIs
4. **Main Loop**: Asynchronous polling loop to update sensor values

## Sensor Types

The service supports various sensor types, each implemented as a separate class:

### RandomSensor

A simple sensor that generates random values (useful for testing).

### ThermalSensor

Fetches temperature readings from a REST API endpoint.

### RPMSensor

Fetches rotational speed readings from a REST API endpoint.

### VoltageSensor

Fetches voltage readings from a REST API endpoint.

### AudioSensorMfcc

Fetches and exposes audio MFCC (Mel-frequency cepstral coefficients) data from a REST API endpoint.

### AudioSensorSpectral

Fetches and exposes audio spectral data from a REST API endpoint.

## OPC UA Configuration

The OPC UA server is configured with:

- **Endpoint**: `opc.tcp://0.0.0.0:4840/freeopcua/server/`
- **Namespace**: `http://examples.freeopcua.github.io`
- **History**: Stored in `plantHistorian.sql` SQLite database

## Data Flow

1. The service initializes an OPC UA server and history storage
2. Sensor adapter classes are instantiated with appropriate API endpoints
3. OPC UA objects and variables are created for each sensor
4. The main loop continuously polls sensor data and updates OPC UA variables
5. Historical data is stored according to configured parameters

## API Integration

The service is designed to integrate with sensor APIs, specifically:

- `http://dmz-sensors/temperature_upper` - Upper thermal sensor
- `http://dmz-sensors/temperature_lower` - Lower thermal sensor
- `http://dmz-sensors/rpm` - RPM sensor
- `http://dmz-sensors/voltage` - Voltage sensor
- `http://dmz-sensors/mfcc` - Audio MFCC data
- `http://dmz-sensors/spectrum` - Audio spectral data

## Advanced Features

### Dynamic Node Discovery

The audio sensor adapters (MFCC and Spectral) dynamically discover available data keys and create corresponding OPC UA nodes. This allows the server to adapt to changes in the underlying data structure.

### Historization

Each sensor value can be historized, with configurable parameters:
- `historize`: Boolean flag to enable/disable historization
- `historize_length`: Maximum number of historical entries to store

### Error Handling

The service includes robust error handling for API connection issues, ensuring continuous operation even when individual sensors are unavailable.

## Dependencies

- `asyncua`: OPC UA server implementation
- `requests`: HTTP client for API communication
- Python 3.12 or higher

## Deployment

The service is containerized using Docker and can be deployed as a standalone container.

### Building the Container

```bash
docker build -t opc-server .
```
```


### Running the Container

```shell script
docker run -p 4840:4840 opc-server
```


## Development

### Project Structure

The service consists of two main files:
- `main.py`: OPC UA server setup and main loop
- `sensors.py`: Sensor adapter classes

### Adding New Sensors

To add a new sensor type:

1. Create a new class in `sensors.py`
2. Implement the required methods (`read_value`)
3. Define sources with appropriate data types
4. Add an instance of your new sensor to the `devices` list in `main.py`

## Connecting to the OPC UA Server

OPC UA clients can connect to the server at:

```
opc.tcp://[server-ip]:4840/freeopcua/server/
```