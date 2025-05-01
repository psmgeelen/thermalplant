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