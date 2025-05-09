import spidev
from .utils import get_logger

logger = get_logger(__name__)

class TempSensor(object):
    """
    A temperature sensor reader class using spidev to access sensors
    connected to SPI bus.
    """

    CELSIUS_PER_BIT = 0.25  # Conversion factor for temperature reading

    def __init__(self, spi_port=1, chip_select=0, max_speed_hz=500000, mode=0b00):
        """
        Initialize the sensor with the specified SPI port and chip select.
        Allows configuring SPI mode and max speed.
        """
        self.spi = spidev.SpiDev()

        # Try opening the desired SPI device
        try:
            self.spi.open(spi_port, chip_select)
        except FileNotFoundError:
            raise ValueError(
                f"SPI port {spi_port} is not available. Check your system configuration."
            )

        self.spi.max_speed_hz = max_speed_hz
        self.spi.mode = mode

        logger.info(
            f"SPI initialized on port {spi_port}, "
            f"chip select {chip_select}, speed {max_speed_hz}Hz, "
            f"mode {mode}"
        )

    def read_temperature(self):
        """
        Read raw values from the sensor and convert to temperature.
        Returns temperature in Celsius (or NaN on error).
        """
        try:
            raw_data = self.spi.xfer2([0x00, 0x00])  # Read 2 bytes
            if not raw_data or len(raw_data) != 2:
                raise ValueError("Failed to read 2 bytes from the sensor.")

            return self._convert_to_temperature(raw_data)
        except Exception as e:
            logger.error(f"Temperature reading error: {str(e)}")
            raise

    def _convert_to_temperature(self, raw_data):
        """
        Convert raw sensor data to temperature in Celsius.
        """
        # Combine the two bytes into a 16-bit number
        combined_data = (raw_data[0] << 8) | raw_data[1]

        # Check for an error signal from the sensor (bit 2 set indicates an error)
        if combined_data & 0x4:
            return float("NaN")

        # Extract the 12-bit temperature data (ignoring the 3 least significant bits)
        temp_data = combined_data >> 3

        # Convert the temperature data to Celsius
        return temp_data * self.CELSIUS_PER_BIT

    def close(self):
        """Close the SPI connection."""
        self.spi.close()
        logger.info("SPI connection closed.")