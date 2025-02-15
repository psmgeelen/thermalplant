from sqlalchemy import create_engine, text
import pandas as pd
import numpy as np
import time
import os


# Step 1: Define connection to PostgreSQL database
db_username = os.environ["POSTGRES_USER"]
db_password = os.environ["POSTGRES_PASSWORD"]
db_host = "database"
db_port = os.environ["POSTGRES_PORT"]
db_database = os.environ["POSTGRES_DB"]

DATABASE_URL = f"postgresql://{db_username}:{db_password}@{db_host}:{db_port}/{db_database}"
print(DATABASE_URL)
engine = create_engine(DATABASE_URL, echo=True) # set echo to true if you need more info on SQL Alchemy.

# Step 2: Define the table creation DDL statement
ddl_statement = """CREATE TABLE sensor_data (
    id SERIAL, -- Autoincremental ID
    timestamp TIMESTAMPTZ NOT NULL, -- Timestamp column
    sensorname TEXT NOT NULL, -- Sensor name
    value DOUBLE PRECISION, -- Sensor value
    PRIMARY KEY (timestamp, sensorname) -- Composite key on timestamp and sensorname
);
"""

ddl_hypertable = """SELECT create_hypertable('sensor_data', 'timestamp', chunk_time_interval => interval '12 hours');"""

retention_policy = """SELECT add_retention_policy('sensor_data', num_items => 100000);"""

# Step 3: Execute the DDL statement
with engine.begin() as connection:
    connection.execute(text(ddl_statement))
    print("Table 'yolo' created successfully (if it didn't already exist).")
    connection.execute(text(ddl_hypertable))
    print("Hypertable 'sensor_data' created successfully.")
    # connection.execute(text(retention_policy))
    # print("Retention policy 'sensor_data' created successfully.")

# Step 4: Use Pandas to generate and insert random data
# Generate random data with Pandas
i = 0
while True:
    data = {"number": [np.random.rand()]}  # Single random float
    df = pd.DataFrame(data)

    # Use Pandas to insert the data into the table
    # df.to_sql("yolo", engine, if_exists="append", index=False, index_label="yomamma")

    print(f"{i}th datapoint inserted: {df}")
    i += 1
    time.sleep(1)
