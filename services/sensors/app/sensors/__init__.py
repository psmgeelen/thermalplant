"""
Sensor package for ThermalPlant monitoring system.
Contains implementations for various sensors:
- Temperature sensor using SPI interface
- RPM sensor using GPIO interface
- Audio processing with PipeWire
"""

# Import sensor classes from individual modules
from .temp import TempSensor
from .rpm import RPMSensor, RPMSensorSettings
from .audio import (
    AudioHandler,
    AudioHandlerSettings,
    PipeWireRecordingLoop,
    AudioProcessingLoop
)

# Define public API for package
__all__ = [
    'TempSensor',
    'RPMSensor', 
    'RPMSensorSettings',
    'AudioHandler',
    'AudioHandlerSettings',
    'PipeWireRecordingLoop',
    'AudioProcessingLoop'
]