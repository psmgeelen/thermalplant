# OPC Scraper Service

A robust data acquisition service that connects to OPC UA servers, retrieves sensor data, and stores it in a TimescaleDB database for time-series analytics.

## Overview

The OPC Scraper service bridges the gap between industrial OPC UA servers and data analytics by continuously collecting sensor data and storing it in an optimized time-series database. It includes automatic database schema management, data retrieval, transformation, and persistent storage capabilities.

## Features

- **OPC UA Integration**: Connects to OPC UA servers and discovers available sensors
- **Hierarchical Node Traversal**: Recursively explores OPC UA node structures to find all sensor data points
- **TimescaleDB Integration**: Stores sensor data in hypertables optimized for time-series data
- **Automatic Table Provisioning**: Creates and configures necessary database tables and jobs
- **Continuous Data Collection**: Periodically polls sensor data and stores it in the database
- **Error Resilience**: Recovers automatically from connection issues and other errors
- **Data Aggregation**: Includes SQL definitions for time-based aggregation views
- **Resource Management**: Implements automatic data retention policies to prevent unbounded growth

## Architecture

The service consists of three main components:

1. **OPC UA Client**: Connects to OPC UA servers and traverses node hierarchies
2. **Database Integration**: Provisions and manages TimescaleDB tables and procedures
3. **Data Processing Pipeline**: Collects, transforms, and stores sensor data

## Data Flow

1. Connect to the OPC UA server and discover available sensor nodes
2. Traverse the node hierarchy to find all Variable nodes containing sensor data
3. Fetch data from each sensor node along with metadata (timestamp, node ID, etc.)
4. Transform the data into a structured format suitable for database storage
5. Insert the data into a TimescaleDB hypertable
6. Continuous aggregation views process the data for different time intervals

## Database Schema

The service uses a TimescaleDB hypertable with the following schema:

- `id`: Auto-incremental ID
- `timestamp`: Timestamp of the reading (TIMESTAMPTZ, partitioning key)
- `nodeid`: OPC UA node ID
- `clientid`: Client identifier
- `sensorname`: Name of the sensor
- `tagname`: Tag name of the specific reading
- `value`: The actual sensor value (DOUBLE PRECISION)

## Continuous Aggregates

The service defines several continuous aggregation views:

- `rolling_avg_sensor_data1m`: 1-minute average aggregation
- `rolling_avg_sensor_data_5s`: 5-second average aggregation
- `rolling_count_sensor_data_5s`: 5-second count aggregation

These views are automatically refreshed on configurable intervals to provide efficient access to aggregated data.

## Resource Management

To prevent unbounded database growth, the service implements:

- A row limiting procedure that keeps the sensor_data table to a maximum of 100,000 rows
- An automated job that runs this procedure every hour
- Continuous aggregate policies that manage the retention of historical aggregated data

## Dependencies

- `asyncua`: Asynchronous OPC UA client
- `sqlalchemy`: Database ORM and query builder
- `pandas`: Data manipulation and transformation
- `psycopg2-binary`: PostgreSQL database driver
- `TimescaleDB`: Time-series database extension for PostgreSQL

## Configuration

The service is configured via environment variables:

- `POSTGRES_USER`: Database username
- `POSTGRES_PASSWORD`: Database password
- `POSTGRES_PORT`: Database port
- `POSTGRES_DB`: Database name

The OPC UA server URL is configurable in the code (default: `opc.tcp://opc_server:4840`).

## Deployment

The service is containerized using Docker and can be deployed as a standalone container:
```
bash
docker build -t opc-scraper .
docker run --env-file .env opc-scraper
```
## Error Handling

The service implements robust error handling:

- Automatic reconnection to the OPC UA server after connection failures
- Graceful handling of missing or inaccessible sensor nodes
- Database transaction management to prevent data corruption
- Comprehensive logging for troubleshooting

## Performance Considerations

- Uses asynchronous I/O for efficient OPC UA communication
- Implements batched database inserts for better performance
- Utilizes TimescaleDB for efficient time-series data storage and retrieval
- Configurable polling intervals to balance between data freshness and system load

