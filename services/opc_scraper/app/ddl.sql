-- Create a continuous aggregate for rolling averages of sensor values
DROP MATERIALIZED VIEW rolling_avg_sensor_data

CREATE MATERIALIZED VIEW rolling_avg_sensor_data
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 minute', timestamp) AS bucketed_time,
    sensorname,
    AVG(value) AS avg_value
FROM sensor_data
GROUP BY bucketed_time, sensorname
ORDER BY bucketed_time ASC
WITH NO DATA;

CALL refresh_continuous_aggregate('rolling_avg_sensor_data', NULL, NULL);


SELECT add_continuous_aggregate_policy('rolling_avg_sensor_data',
  start_offset => NULL,
  end_offset => INTERVAL '1 minute,',
  schedule_interval => INTERVAL '1 minute');

SELECT add_continuous_aggregate_policy('rolling_avg_sensor_data',
  start_offset => INTERVAL '10 minutes',
  end_offset => INTERVAL '30 seconds',
  schedule_interval => INTERVAL '30 seconds');