# test/test_temp.py
import pytest
import math
from unittest.mock import patch, MagicMock

# No need for sys.path manipulation since conftest.py handles it
from app.sensors.temp import TempSensor


# Mock the hardware dependencies
@pytest.fixture
def mock_spidev():
    with patch('app.sensors.temp.spidev') as mock:
        # Create a mock for SpiDev
        mock_spi = MagicMock()

        # Configure the open method
        mock_spi.open = MagicMock()

        # Configure xfer2 to return test data representing 25°C
        mock_spi.xfer2.return_value = [0x03, 0x20]

        # Set up the SpiDev constructor to return our mock
        mock.SpiDev.return_value = mock_spi

        yield mock


@pytest.fixture
def mock_logger():
    with patch('app.sensors.temp.logger') as mock:
        yield mock

def test_temp_sensor_init(mock_spidev, mock_logger):
    """Test that the temperature sensor initializes correctly."""
    # Arrange & Act
    sensor = TempSensor(spi_port=1, chip_select=0)
    
    # Assert
    mock_spidev.SpiDev.assert_called_once()
    sensor.spi.open.assert_called_once_with(1, 0)
    sensor.spi.max_speed_hz = 500000
    sensor.spi.mode = 0
    mock_logger.info.assert_called()

def test_temp_sensor_read_temperature(mock_spidev, mock_logger):
    """Test that read_temperature correctly processes sensor data."""
    # Arrange
    sensor = TempSensor()
    # Configure mock to return data representing 25°C (0x0320 >> 3 * 0.25 = 25)
    sensor.spi.xfer2.return_value = [0x03, 0x20]
    
    # Act
    temperature = sensor.read_temperature()
    
    # Assert
    sensor.spi.xfer2.assert_called_with([0x00, 0x00])
    assert temperature == 25.0

def test_temp_sensor_read_error(mock_spidev, mock_logger):
    """Test that read_temperature handles error conditions."""
    # Arrange
    sensor = TempSensor()
    # Set up xfer2 to return error data (bit 2 set)
    sensor.spi.xfer2.return_value = [0x00, 0x04]  # Error bit set
    
    # Act
    temperature = sensor.read_temperature()
    
    # Assert
    assert math.isnan(temperature)

def test_temp_sensor_close(mock_spidev, mock_logger):
    """Test that the sensor can be closed properly."""
    # Arrange
    sensor = TempSensor()
    
    # Act
    sensor.close()
    
    # Assert
    sensor.spi.close.assert_called_once()
    mock_logger.info.assert_called_with("SPI connection closed.")