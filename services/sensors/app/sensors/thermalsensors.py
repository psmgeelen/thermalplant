import RPi
class ThermalSensorBase(object):
    """source: https://github.com/tdack/MAX6675/blob/master/MAX6675/MAX6675.py"""
    

	def __init__(self, clk=None, cs=None, do=None, spi=None, gpio=None):
		"""Initialize MAX6675 device with software SPI on the specified CLK,
		CS, and DO pins.  Alternatively can specify hardware SPI by sending an
		Adafruit_GPIO.SPI.SpiDev device in the spi parameter.
		"""
        self.name = "thermalSensor" + prefix
        self.gpio_pin = gpio_pin
        self.sources = [
            {
                "name": "random",
                "datatype": ua.Variant(0.0, ua.VariantType.Double),
                "historize": True,
                "historize_length": 1000,
                "func": self.read_value,
                "writable": True
            },
        ]
		self._logger = logging.getLogger('MAX6675.MAX6675')
		self._spi = None
		# Handle hardware SPI
		if spi is not None:
			self._logger.debug('Using hardware SPI')
			self._spi = spi
		elif clk is not None and cs is not None and do is not None:
			self._logger.debug('Using software SPI')
			# Default to platform GPIO if not provided.
			if gpio is None:
				gpio = GPIO.get_platform_gpio()
			self._spi = SPI.BitBang(gpio, clk, None, do, cs)
		else:
			raise ValueError('Must specify either spi for for hardware SPI or clk, cs, and do for softwrare SPI!')
		self._spi.set_clock_hz(5000000)
		self._spi.set_mode(0)
		self._spi.set_bit_order(SPI.MSBFIRST)

	def read_value(self):
		"""Return the thermocouple temperature value in degrees celsius."""
		v = self._read16()
		# Check for error reading value.
		if v & 0x4:
			return float('NaN')
		# Check if signed bit is set.
		if v & 0x80000000:
			# Negative value, take 2's compliment. Compute this with subtraction
			# because python is a little odd about handling signed/unsigned.
			v >>= 3 # only need the 12 MSB
			v -= 4096
		else:
			# Positive value, just shift the bits to get the value.
			v >>= 3 # only need the 12 MSB
		# Scale by 0.25 degrees C per bit and return value.
		return v * 0.25

	def _read16(self):
		# Read 16 bits from the SPI bus.
		raw = self._spi.read(2)
		if raw is None or len(raw) != 2:
			raise RuntimeError('Did not read expected number of bytes from device!')
		value = raw[0] << 8 | raw[1]
		self._logger.debug('Raw value: 0x{0:08X}'.format(value & 0xFFFFFFFF))
		return value
    