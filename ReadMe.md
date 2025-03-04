# Thermal Mini Plant

## Scope
This project aims to showcase a predictive maintenance solution for a thermal-based manufacturing plant. For all the service we deploy a OPC-UA server that we can pull metrics from. We collect the data in a Postgress TimescaleDB database and visualize the results in Grafana. Finally we use some simple ML to highlight Failure Modes. 

## Components
* Infra
  * Grafana
  * Postgress TimescaleDB
  * Services
* Services
  * Written in Python
  * Dependencies are managed with Poetry
  * Services are based on the OPC-UA Architecture
* Hardware
  * Raspberry Pi 5
  * Thermal sensors:
  * RPM sensor: 
* Automation
  * Taskfiles

## Quickstart

## CI

## Resources
* OPC-UA Library: 
  * Repo: https://github.com/FreeOpcUa/opcua-asyncio
  * Documentation: https://opcua-asyncio.readthedocs.io/en/latest/

Note to self: install toolkit on high available DB
```postgresql
CREATE EXTENSION IF NOT EXISTS timescaledb_toolkit;
```