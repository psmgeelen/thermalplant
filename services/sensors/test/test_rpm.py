# test/test_rpm.py
import pytest
import time
from unittest.mock import patch, MagicMock
from collections import deque
import math

# No need for sys.path manipulation since conftest.py handles it
from app.sensors.rpm import RPMSensor, RPMSensorSettings


# Mock the hardware dependencies
@pytest.fixture
def mock_gpiozero():
    with patch('app.sensors.rpm.gpiozero') as mock:
        # Create a mock for InputDevice
        mock_input_device = MagicMock()
        # Configure the value property to alternate between 0 and 1
        mock_input_device.value.side_effect = [0, 1] * 100

        # Set up the InputDevice constructor to return our mock
        mock.InputDevice.return_value = mock_input_device

        # Mock the LGPIOFactory
        mock_lgpio_factory = MagicMock()
        mock.pins.lgpio.LGPIOFactory.return_value = mock_lgpio_factory

        yield mock


@pytest.fixture
def mock_threading():
    with patch('app.sensors.rpm.threading') as mock:
        yield mock


@pytest.fixture
def mock_logger():
    with patch('app.sensors.rpm.logger') as mock:
        yield mock

@pytest.fixture
def mock_threading():
    with patch('app.sensors.rpm.threading') as mock:
        # Create a mock for Thread
        mock_thread = MagicMock()
        # Configure Thread constructor to return our mock
        mock.Thread.return_value = mock_thread
        yield mock

def test_rpm_sensor_init(mock_gpiozero, mock_threading, mock_logger):
    """Test that RPM sensor initializes correctly."""
    # Arrange & Act
    sensor = RPMSensor(
        gpio_pin=22, 
        measurement_interval=0.001, 
        measurement_window=100, 
        sample_size=8
    )
    
    # Assert
    # InputDevice is created inside the _do_measurement method, not directly in __init__
    # So we should check that Thread was started with the right target
    assert sensor.running is True
    assert sensor.gpiopin == 22
    assert sensor.measurement_interval == 0.001
    assert sensor.measurement_window == 100
    assert sensor.sample_size == 8
    assert sensor.measurements.maxlen == 100
    
    # Check that the Thread was created and started
    mock_threading.Thread.assert_called_once()
    # Get the Thread instance that was created
    thread_instance = mock_threading.Thread.return_value
    # Check that start was called on that instance
    thread_instance.start.assert_called_once()

def test_rpm_sensor_read_empty(mock_gpiozero, mock_threading, mock_logger):
    """Test RPM calculation with empty measurements."""
    # Arrange
    sensor = RPMSensor(
        gpio_pin=22, 
        measurement_interval=0.001, 
        measurement_window=100, 
        sample_size=8
    )
    sensor.measurements.clear()  # Ensure measurements are empty
    
    # Act
    rpm = sensor.read_rpm()
    
    # Assert
    assert rpm == 0

def test_rpm_sensor_read_with_data(mock_gpiozero, mock_threading, mock_logger):
    """Test RPM calculation with sample data."""
    # Arrange
    sensor = RPMSensor(
        gpio_pin=22, 
        measurement_interval=0.001, 
        measurement_window=100, 
        sample_size=8
    )
    
    # Simulate measurements for exactly one revolution (8 state changes)
    # Each state change is 10ms apart, so one revolution takes 80ms
    # 60 seconds / 0.08 seconds = 750 RPM
    base_time = time.time()
    base_time_ns = time.time_ns()
    
    # Fill with data for a complete revolution (8 entries for 4 blades)
    for i in range(8):
        sensor.measurements.append({
            "state": i % 2,  # Alternating 0 and 1
            "time": base_time + (i * 0.01),  # 10ms intervals
            "time_ns": base_time_ns + (i * 10_000_000)  # 10ms in nanoseconds
        })
    
    # Act
    rpm = sensor.read_rpm()
    
    # Assert
    assert 0 <= rpm <= 500  # Allow small margin for floating point precision

def test_rpm_sensor_stop(mock_gpiozero, mock_threading, mock_logger):
    """Test that the RPM sensor can be stopped."""
    # Arrange
    sensor = RPMSensor(
        gpio_pin=22, 
        measurement_interval=0.001, 
        measurement_window=100, 
        sample_size=8
    )
    
    # Act
    sensor.stop()
    
    # Assert
    assert sensor.running is False
    sensor.measurement_thread.join.assert_called_once()