-- Create a continuous aggregate for rolling averages of sensor values
DROP MATERIALIZED VIEW rolling_avg_sensor_data1m

CREATE MATERIALIZED VIEW rolling_avg_sensor_data1m
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 minute', timestamp) AS bucketed_time,
    sensorname,
    tagname,
    AVG(value) AS avg_value
FROM sensor_data
GROUP BY bucketed_time, sensorname, tagname
ORDER BY bucketed_time ASC
WITH NO DATA;

CALL refresh_continuous_aggregate('rolling_avg_sensor_data1m', NULL, NULL);


SELECT add_continuous_aggregate_policy('rolling_avg_sensor_data1m',
  start_offset => INTERVAL '10 minutes',
  end_offset => INTERVAL '30 seconds',
  schedule_interval => INTERVAL '30 seconds');

-- Create a continuous aggregate for rolling averages of sensor values
DROP MATERIALIZED VIEW rolling_avg_sensor_data_5s

CREATE MATERIALIZED VIEW rolling_avg_sensor_data_5s
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('5 seconds', timestamp) AS bucketed_time,
    sensorname,
    tagname,
    AVG(value) AS avg_value
FROM sensor_data
GROUP BY bucketed_time, sensorname, tagname
ORDER BY bucketed_time ASC
WITH NO DATA;

CALL refresh_continuous_aggregate('rolling_avg_sensor_data_5s', NULL, NULL);


SELECT add_continuous_aggregate_policy('rolling_avg_sensor_data_5s',
  start_offset => INTERVAL '10 minutes',
  end_offset => INTERVAL '5 seconds',
  schedule_interval => INTERVAL '5 seconds');


####
DROP MATERIALIZED VIEW rolling_count_sensor_data_5s

CREATE MATERIALIZED VIEW rolling_count_sensor_data_5s
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('5 seconds', timestamp) AS bucketed_time,
    sensorname,
    tagname,
    AVG(value) AS counted_values
FROM sensor_data
GROUP BY bucketed_time, sensorname, tagname
ORDER BY bucketed_time ASC
WITH NO DATA;

CALL refresh_continuous_aggregate('rolling_count_sensor_data_5s', NULL, NULL);


SELECT add_continuous_aggregate_policy('rolling_count_sensor_data_5s',
  start_offset => INTERVAL '10 minutes',
  end_offset => INTERVAL '5 seconds',
  schedule_interval => INTERVAL '5 seconds');
