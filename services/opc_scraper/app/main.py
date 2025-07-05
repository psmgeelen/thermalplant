import logging
from sqlalchemy import create_engine, text
import pandas as pd
# import numpy as np
import time
import os
from asyncua import Client, ua
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class Scraper(object):
    """
    Manages the integration with OPC UA server and a PostgreSQL database, allowing for data fetching
    from sensors and provisioning related database tables. The class provides utilities to execute
    SQL queries, check table existence, verify hypertable status, and manage database job scheduling.

    :ivar OPC_SERVER_URL: The URL of the OPC UA server.
    :type OPC_SERVER_URL: str
    :ivar db_username: The username for the PostgreSQL database.
    :type db_username: str
    :ivar db_password: The password for the PostgreSQL database.
    :type db_password: str
    :ivar db_host: The hostname or IP address of the PostgreSQL database server.
    :type db_host: str
    :ivar db_port: The port number for the PostgreSQL database server.
    :type db_port: str
    :ivar db_database: The database name to connect to in the PostgreSQL server.
    :type db_database: str
    :ivar DATABASE_URL: The full connection string formatted to connect to the database.
    :type DATABASE_URL: str
    :ivar engine: SQLAlchemy engine for executing database operations.
    :type engine: sqlalchemy.engine.Engine
    :ivar sensor_nodes: A dictionary containing sensor node information. Each key represents
        an item number, and its value contains a dictionary with node details such as "nodeId"
        and "name".
    :type sensor_nodes: dict[int, dict[str, str]]
    """
    def __init__(self, OPC_SERVER_URL: str, db_username: str, db_password: str, db_host: str, db_port: str, db_database: str):
        self.OPC_SERVER_URL = OPC_SERVER_URL
        self.db_username = db_username
        self.db_password = db_password
        self.db_host = db_host
        self.db_port = db_port
        self.db_database = db_database
        self.DATABASE_URL = f"postgresql://{db_username}:{db_password}@{db_host}:{db_port}/{db_database}"
        self.engine = create_engine(self.DATABASE_URL, echo=True)
        self.sensor_nodes = {}

    @staticmethod
    def _table_exists(connection, table_name) -> bool | None:
        """
        Checks if a table exists in the connected database by querying the
        information schema of the database. It executes a query to determine
        if a table with the given name exists and returns the result.

        :param connection: A database connection object that allows execution
                           of SQL queries.
        :type connection: Any
        :param table_name: The name of the table whose existence is to be
                           checked.
        :type table_name: str

        :return: Returns a boolean value indicating whether the table exists,
                 or None if the query result cannot be fetched.
        :rtype: bool | None
        """
        exists_check_query = """
        SELECT EXISTS (
            SELECT FROM information_schema.tables
            WHERE table_name = :table_name
        );
        """
        result = connection.execute(text(exists_check_query), {"table_name": table_name}).scalar()
        return result


    @staticmethod
    def _is_hypertable(connection, table_name) -> bool | None:
        """
        Determines if the specified table is a hypertable in TimescaleDB. This function
        executes a query against the TimescaleDB's `timescaledb_information.hypertables`
        view to check the existence of the provided table name.

        The function returns True if the table is a hypertable, False if not, or None
        if no result is returned or the connection fails to retrieve the information.

        :param connection: A database connection object used to execute the query.
        :param table_name: The name of the table to check for hypertable status.
            It should be a valid string that matches the table name in the database.
        :return: A boolean indicating whether the table is a hypertable (True or False),
            or None if the query fails to return a result.
        :rtype: bool | None
        """
        hypertable_check_query = """
        SELECT EXISTS (
            SELECT 1
            FROM timescaledb_information.hypertables
            WHERE hypertable_name = :table_name
        );
        """
        result = connection.execute(text(hypertable_check_query), {"table_name": table_name}).scalar()
        return result


    @staticmethod
    def _job_exists(connection, job_name) -> bool | None:
        """
        Checks whether a job with the given name exists in the 'timescaledb_information.jobs' table.

        This function executes a query to verify if there is an entry in the table
        'jobs' under the schema 'timescaledb_information' where 'proc_name' matches
        the provided job name. It uses the provided database connection object for
        executing the query.

        :param connection: The database connection object used to execute the query.
        :type connection: Any

        :param job_name: The name of the job to check in the database.
        :type job_name: str

        :return: A boolean indicating whether the job exists, or `None` if the query
                 fails or no result is found.
        :rtype: bool | None
        """
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
    def provision_table(self):
        """
        Provisions and manages a database table called "sensor_data" with associated constraints
        and maintenance processes. This function ensures the table exists, converts it
        to a hypertable if required, configures a procedure to limit its row count,
        and schedules the procedure to automatically execute periodically.

        The following operations are performed:
        1. Check for the existence of the table and create it if needed.
        2. Convert the table into a TimescaleDB hypertable if it is not already.
        3. Create or replace a PostgreSQL procedure that limits the number of rows
           in the table to a maximum of 100,000.
        4. Schedule the row limiting procedure to run every hour if there isn't
           an existing scheduled job.

        :param engine: Database connection engine used to interact with the database.
        :type engine: sqlalchemy.engine.Engine
        :param table_exists: Function that checks if a table exists in the database.
        :type table_exists: Callable[[Connection, str], bool]
        :param is_hypertable: Function that checks if a table is a TimescaleDB hypertable.
        :type is_hypertable: Callable[[Connection, str], bool]
        :param job_exists: Function that checks if a maintenance job exists in the database.
        :type job_exists: Callable[[Connection, str], bool]
        :param logger: Logger instance to output messages during the provisioning process.
        :type logger: logging.Logger

        :raises RuntimeError: If any critical operation (e.g., creating a hypertable or scheduling a job)
            fails, a RuntimeError will be raised to indicate the failure.
        :return: None
        """
        ddl_statement = """CREATE TABLE IF NOT EXISTS sensor_data (
            id SERIAL, -- Auto-incremental ID
            timestamp TIMESTAMPTZ NOT NULL, -- Timestamp column
            nodeid TEXT NOT NULL, -- Node ID
            clientid TEXT NOT NULL, -- Client ID
            sensorname TEXT NOT NULL, -- Sensor name
            tagname TEXT, -- Tag name
            value DOUBLE PRECISION, -- Sensor value
            PRIMARY KEY (timestamp, sensorname) -- Composite key on timestamp and sensorname
        );"""
        ddl_hypertable = """SELECT create_hypertable('sensor_data', 'timestamp', chunk_time_interval => interval '5 mins', migrate_data => true);"""

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

        with self.engine.begin() as connection:
            # Check if the table exists
            if not self._table_exists(connection, table_name):
                logger.warning(f"Table '{table_name}' does not exist. Creating it...")
                connection.execute(text(ddl_statement))
                logger.warning(f"Table '{table_name}' created successfully.")
            else:
                logger.info(f"Table '{table_name}' already exists.")

            # Check if the table is a hypertable
            if not self._is_hypertable(connection, table_name):
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
            if not self._job_exists(connection, job_name):
                logger.warning("No scheduled job found for row limiting. Scheduling it...")
                try:
                    connection.execute(text(schedule_job))
                    logger.warning("Scheduled row limiting job successfully.")
                except Exception as e:
                    logger.error(f"Failed to schedule job: {str(e)}")
                    raise RuntimeError("Failed to schedule job.") from e
            else:
                logger.info("Row limiting job is already scheduled.")

    async def fetch_sensor_data(self, OPC_SERVER_URL: str, sensor_nodes: dict[int, dict[str, str]]) -> list[dict[
        str, str | pd.Timestamp]] | None:
        """
        Fetch sensor data asynchronously from an OPC UA server for specified sensor nodes.

        This method connects to the specified OPC UA server and retrieves data from the
        given sensor nodes. Each sensor node is defined by its unique identifier
        (`nodeId`) and name. Using recursion, it explores deeper nodes within the
        hierarchy, fetching values only from nodes of the 'Variable' NodeClass.
        Errors are logged, and the fetching continues for remaining nodes.

        :param OPC_SERVER_URL: The URL of the OPC UA server to connect to.
        :type OPC_SERVER_URL: str
        :param sensor_nodes: A dictionary where the keys are integers representing
            node indexes and the values are dictionaries containing the node ID
            (``nodeId``) and name (``name``) of each sensor node.
        :type sensor_nodes: dict[int, dict[str, str]]
        :return: A list of dictionaries containing the timestamp, sensor name,
            and the fetched value for each sensor node; or None if the operation
            fails.
        :rtype: list[dict[str, str | pd.Timestamp]] | None
        """
        async with Client(OPC_SERVER_URL) as client:
            try:
                logger.info(f"Connected to OPC UA server: {OPC_SERVER_URL}")
                aggregated_data = []

                for itemnr, sensor_node in sensor_nodes.items():
                    try:
                        if 'sensor' in sensor_node['name'].lower():
                            logger.info(
                            f"Fetching data for node {itemnr}: {sensor_node['nodeId']} with name: {sensor_node['name']}")
                            node = client.get_node(sensor_node["nodeId"])

                            # Initiate recursive fetch process
                            fetched_data = await self.traverse_and_fetch(node, client, sensor_node["name"])
                            aggregated_data.extend(fetched_data)
                        else:
                            logger.info(
                                f"Is not a sensor, skipping {itemnr}: {sensor_node['nodeId']} with name: {sensor_node['name']}")


                    except Exception as sensor_error:
                        logger.error(
                            f"Error fetching data for sensor {sensor_node['name']} ({sensor_node['nodeId']}): {sensor_error}")
                        continue

                return aggregated_data
            except Exception as e:
                logger.error(f"Failed to fetch sensor data from OPC UA server: {e}")
                return []

    async def traverse_and_fetch(self, node, client, sensor_name) -> dict[str, str | pd.Timestamp] | None:
        """
        Recursively fetch data from nodes, traversing the hierarchy. Only fetches data from Variable nodes.

        :param node: The current node to analyze.
        :param client: The OPC UA client instance.
        :param sensor_name: Name of the sensor for logging purposes.
        :return: List of fetched data records.
        """
        data = []
        try:
            node_class = await node.read_node_class()

            if node_class == ua.NodeClass.Variable:
                try:
                    value = await node.read_value()
                    timestamp = pd.Timestamp.now()
                    logger.info(
                        f"Fetched data from {sensor_name} (NodeId: {node.nodeid.to_string()}): {value} at {timestamp}")

                    # Get the tag name (browse name) of the node
                    browse_name = await node.read_browse_name()
                    tag_name = browse_name.Name
                    
                    data.append({
                        "timestamp": timestamp,
                        "nodeid": node.nodeid.to_string(),
                        "clientid": str(client),
                        "sensorname": str(sensor_name),
                        "tagname": tag_name,
                        "value": value
                    })
                except Exception as value_error:
                    logger.error(
                        f"Error reading value for NodeId {node.nodeid.to_string()} of sensor {sensor_name}: {value_error}")

            elif node_class == ua.NodeClass.Object:
                try:
                    children = await node.get_children()
                    for child in children:
                        child_data = await self.traverse_and_fetch(child, client, sensor_name)
                        data.extend(child_data)
                except Exception as traverse_error:
                    logger.error(
                        f"Error traversing Object NodeId {node.nodeid.to_string()} for sensor {sensor_name}: {traverse_error}")
        except Exception as node_error:
            logger.error(
                f"Error reading NodeClass for NodeId {node.nodeid.to_string()} of sensor {sensor_name}: {node_error}")

        return data

    @staticmethod
    async def fetch_sensors(OPC_SERVER_URL: str) -> dict[int, dict[str, str]]:
        """
        Fetch information about sensors connected to the OPC UA server.

        This function establishes a connection to the specified OPC UA server and retrieves a
        list of sensor nodes. It parses the connected nodes and filters out only nodes of type
        `Variable`. Each detected sensor node is then added to a dictionary with relevant
        information such as node ID and display name. The function also logs the connections made,
        sensor information, and any errors encountered during execution.

        :param OPC_SERVER_URL: The URL of the OPC UA server to connect to.
        :type OPC_SERVER_URL: str

        :return: A dictionary where keys are the indices of the variable nodes, and values are the
            corresponding details of each sensor. Each value is a dictionary containing the
            "nodeId" and "name" of a sensor node.
        :rtype: dict[int, dict[str, str]]
        """
        async with Client(OPC_SERVER_URL) as client:
            try:
                logger.info(f"Connected to OPC UA server: {OPC_SERVER_URL}")
                root_node = client.nodes.objects
                sensors = await root_node.get_children()

                sensor_nodes = {}
                count = 0
                for sensor in sensors:
                    display_name = await sensor.read_display_name()
                    node_class = await sensor.read_node_class()
                    node_id = sensor.nodeid

                    # Print all node attributes for debugging
                    logger.info(
                        f"Node: {node_id}, DisplayName: {display_name.Text}, NodeClass: {node_class.name}, NamespaceIndex: {node_id.NamespaceIndex}")

                    sensor_nodes[count] = {
                        "nodeId": sensor.nodeid,
                        "name": display_name.Text
                    }
                    await asyncio.sleep(0.4) ## Polling rate
                    count += 1

                logger.info(f"Detected sensors: {sensor_nodes}")
                return sensor_nodes
            except Exception as e:
                logger.error(f"Failed to fetch sensors from OPC UA server: {e}")
                return {}


    # Modified data insertion workflow
    async def insert_data(self):
        """
        Continuously fetch sensor data from the OPC UA server and insert it
        into the database. This ensures that sensor data is retrieved and stored
        periodically for monitoring or analysis purposes.

        :return: None
        :rtype: None
        :raises Exception: If an error occurs during data insertion into the database.
        """
        sensor_nodes = await self.fetch_sensors(OPC_SERVER_URL=self.OPC_SERVER_URL)
        if not sensor_nodes:
            logger.error("No sensors detected. Exiting.")
            return

        while True:
            data_rows = await self.fetch_sensor_data(OPC_SERVER_URL=self.OPC_SERVER_URL, sensor_nodes=sensor_nodes)
            if not data_rows:
                logger.warning("No data fetched in this cycle. Retrying...")
                time.sleep(5)
                continue

            df = pd.DataFrame(data_rows)
            try:
                df.to_sql("sensor_data", self.engine, if_exists="append", index=False)
                logger.warning(f"Inserted {len(df)} datapoints into the database.")
            except Exception as e:
                logger.error(f"Error inserting data into the database: {e}")
                break

            time.sleep(1)  # Periodic fetch rate


if __name__ == "__main__":
    scraper = Scraper(
        OPC_SERVER_URL="opc.tcp://opc_server:4840",  # Replace with the actual URL of your OPC UA server
        db_username=os.environ["POSTGRES_USER"],
        db_password=os.environ["POSTGRES_PASSWORD"],
        db_host="database",
        db_port=os.environ["POSTGRES_PORT"],
        db_database=os.environ["POSTGRES_DB"]
    )
    scraper.provision_table()
    time.sleep(5)

    while True:
        try:
            asyncio.run(scraper.insert_data())
        except Exception as e:
            logger.critical(f"Critical error: {str(e)}")

        logger.info("Restarting scrape in 3 seconds...")
        time.sleep(3)

