import logging
from sqlalchemy import create_engine, text
import pandas as pd
# import numpy as np
import time
import os
from asyncua import Client, Node
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Step 1: Define connection to PostgreSQL database
db_username = os.environ["POSTGRES_USER"]
db_password = os.environ["POSTGRES_PASSWORD"]
db_host = "database"
db_port = os.environ["POSTGRES_PORT"]
db_database = os.environ["POSTGRES_DB"]
DATABASE_URL = f"postgresql://{db_username}:{db_password}@{db_host}:{db_port}/{db_database}"
engine = create_engine(DATABASE_URL, echo=True)  # Set echo to true for SQLAlchemy debug information.

# OPC UA server setup
OPC_SERVER_URL = "opc.tcp://opc_server:4840"  # Replace with the actual URL of your OPC UA server



# Function to check if a table exists
def table_exists(connection, table_name):
    exists_check_query = """
    SELECT EXISTS (
        SELECT FROM information_schema.tables
        WHERE table_name = :table_name
    );
    """
    result = connection.execute(text(exists_check_query), {"table_name": table_name}).scalar()
    return result


# Function to check if a table is a hypertable
def is_hypertable(connection, table_name):
    hypertable_check_query = """
    SELECT EXISTS (
        SELECT 1
        FROM timescaledb_information.hypertables
        WHERE hypertable_name = :table_name
    );
    """
    result = connection.execute(text(hypertable_check_query), {"table_name": table_name}).scalar()
    return result


# Function to check if a TimescaleDB job already exists
def job_exists(connection, job_name):
    job_check_query = """
    SELECT EXISTS (
        SELECT *
        FROM timescaledb_information.jobs
        WHERE proc_name = :job_name
    );
    """
    result = connection.execute(text(job_check_query), {"job_name": job_name}).scalar()
    return result


# Main provisioning workflow
def provision_table():
    ddl_statement = """CREATE TABLE IF NOT EXISTS sensor_data (
        id SERIAL, -- Auto-incremental ID
        timestamp TIMESTAMPTZ NOT NULL, -- Timestamp column
        sensorname TEXT NOT NULL, -- Sensor name
        value DOUBLE PRECISION, -- Sensor value
        PRIMARY KEY (timestamp, sensorname) -- Composite key on timestamp and sensorname
    );"""
    ddl_hypertable = """SELECT create_hypertable('sensor_data', 'timestamp', chunk_time_interval => interval '12 hours');"""

    # Create procedure to enforce max 100,000 rows
    create_procedure = """
    CREATE OR REPLACE PROCEDURE limit_table_rows()
    LANGUAGE plpgsql
    AS $$
    BEGIN
        DELETE FROM sensor_data
        WHERE ctid IN (
            SELECT ctid
            FROM sensor_data
            ORDER BY timestamp
            LIMIT GREATEST(0, (SELECT COUNT(*) FROM sensor_data) - 100000)
        );
    END;
    $$;
    """

    # Schedule the procedure to run every 1 hour
    schedule_job = """
    SELECT add_job('limit_table_rows', '1 hour');
    """

    table_name = "sensor_data"
    job_name = "limit_table_rows"

    with engine.begin() as connection:
        # Check if the table exists
        if not table_exists(connection, table_name):
            logger.warning(f"Table '{table_name}' does not exist. Creating it...")
            connection.execute(text(ddl_statement))
            logger.warning(f"Table '{table_name}' created successfully.")
        else:
            logger.info(f"Table '{table_name}' already exists.")

        # Check if the table is a hypertable
        if not is_hypertable(connection, table_name):
            logger.warning(f"Table '{table_name}' is not a hypertable. Converting it to a hypertable...")
            try:
                connection.execute(text(ddl_hypertable))
                logger.warning(f"Hypertable '{table_name}' created successfully.")
            except Exception as e:
                logger.error(f"Failed to create hypertable: {str(e)}")
                raise RuntimeError("Failed to create hypertable.") from e
        else:
            logger.info(f"Table '{table_name}' is already a hypertable.")

        # Create procedure to limit rows
        logger.warning("Creating or replacing the procedure to enforce a max row limit...")
        try:
            connection.execute(text(create_procedure))
            logger.warning("Row limiting procedure created successfully.")
        except Exception as e:
            logger.error(f"Failed to create procedure: {str(e)}")
            raise

        # Schedule the job if it doesn't exist
        if not job_exists(connection, job_name):
            logger.warning("No scheduled job found for row limiting. Scheduling it...")
            try:
                connection.execute(text(schedule_job))
                logger.warning("Scheduled row limiting job successfully.")
            except Exception as e:
                logger.error(f"Failed to schedule job: {str(e)}")
                raise RuntimeError("Failed to schedule job.") from e
        else:
            logger.info("Row limiting job is already scheduled.")

async def fetch_sensors():
    """Connect to the OPC UA server and list all sensors dynamically."""
    async with Client(OPC_SERVER_URL) as client:
        try:
            logger.info(f"Connected to OPC UA server: {OPC_SERVER_URL}")
            # Browse the root node and identify all sensors (modify based on your server configuration)
            root_node = client.nodes.objects
            sensors = await root_node.get_children()

            sensor_nodes = {}
            count = 0
            for sensor in sensors:
                sensor_name = await sensor.read_display_name()

                sensor_nodes[count] = {
                    "nodeId": sensor.nodeid,
                    "name": sensor_name.Text
                }
                count += 1

            logger.info(f"Detected sensors: {sensor_nodes}")
            return sensor_nodes
        except Exception as e:
            logger.error(f"Failed to fetch sensors from OPC UA server: {e}")
            return {}

# Function to fetch sensor values dynamically
async def fetch_sensor_data(sensor_nodes):
    """
    Fetch data from all detected sensors.
    :param sensor_nodes: Dictionary of sensor node IDs and their names.
    """
    async with Client(OPC_SERVER_URL) as client:
        try:
            logger.info(f"Connected to OPC UA server: {OPC_SERVER_URL}")
            data = []
            for itemnr, sensor_node in sensor_nodes.items():
                try:
                    logger.info(f"Fetching data for node: {sensor_node["nodeId"]} with name: {sensor_node["name"]}")

                    node = client.get_node(sensor_node["nodeId"])
                    value = await node.read_value()
                    timestamp = pd.Timestamp.now()
                    logger.info(f"Fetched data from {sensor_node["name"]} ({sensor_node["nodeId"]}): {value} at {timestamp}")

                    data.append({"timestamp": timestamp, "sensorname": sensor_node["name"], "value": value})
                except Exception as sensor_error:
                    logger.error(f"Error fetching data for sensor {sensor_node["name"]} ({sensor_node["nodeId"]}): {sensor_error}")
                    continue

            return data
        except Exception as e:
            logger.error(f"Failed to fetch sensor data from OPC UA server: {e}")
            return []

# Modified data insertion workflow
async def insert_data():
    """
    Continuously fetch data from the OPC UA server and insert it into the database.
    """
    sensor_nodes = await fetch_sensors()
    if not sensor_nodes:
        logger.error("No sensors detected. Exiting.")
        return

    while True:
        data_rows = await fetch_sensor_data(sensor_nodes)
        if not data_rows:
            logger.warning("No data fetched in this cycle. Retrying...")
            time.sleep(5)
            continue

        df = pd.DataFrame(data_rows)
        try:
            df.to_sql("sensor_data", engine, if_exists="append", index=False)
            logger.warning(f"Inserted {len(df)} datapoints into the database.")
        except Exception as e:
            logger.error(f"Error inserting data into the database: {e}")
            break

        time.sleep(1)  # Periodic fetch rate


if __name__ == "__main__":


    try:
        provision_table()
        asyncio.run(insert_data())
    except Exception as e:
        logger.critical(f"Critical error: {str(e)}")

