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