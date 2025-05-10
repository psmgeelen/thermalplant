import time
import threading
from collections import deque
import board
import busio
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn
from pydantic import BaseModel, Field, validator
from .utils import get_logger

logger = get_logger(__name__)

class VoltageSensor(object):

    def __init__(
            self,
            measurement_interval: float,
            measurement_window: int,
            sample_size: int,
            i2c_address: int = 0x48,
            adc_channel: int = 0,
    ):
        self.i2c_address = i2c_address
        self.adc_channel = adc_channel
        self.measurement_interval = measurement_interval
        self.measurement_window = measurement_window
        self.sample_size = sample_size
        self.gain = 2/3  # Start with the lowest gain to avoid overloading the ADC
        self.max_voltage = 6.144  # Default max voltage for gain 2/3
        self.measurements = deque(maxlen=self.measurement_window)
        self.running = False  # Flag to control the measurement thread
        self._start_measurement_thread()

    def _adjust_gain(self, last_voltage):
        """
        Adjust the gain dynamically based on the most recent measurement.
        If the voltage is above 80% of the current max voltage, zoom out (decrease gain).
        If the voltage is too low (below 80%), zoom in (increase gain).
        """
        # Gain settings and their corresponding max voltage ranges
        gain_settings = {
            2/3: 6.144,  # +/-6.144V
            1: 4.096,    # +/-4.096V
            2: 2.048,    # +/-2.048V
            4: 1.024,    # +/-1.024V
            8: 0.512,    # +/-0.512V
            16: 0.256    # +/-0.256V
        }

        # Determine the 80% threshold
        threshold = 0.8 * self.max_voltage

        # Zoom in or out based on the measurement
        if last_voltage > threshold:
            # Zoom out (lower gain)
            for gain, voltage_range in sorted(gain_settings.items(), reverse=True):
                if voltage_range >= last_voltage:
                    self.gain = gain
                    self.max_voltage = voltage_range
                    logger.info(f"Zoomed out: new gain is {self.gain}x")
                    break
        elif last_voltage < threshold:
            # Zoom in (increase gain)
            for gain, voltage_range in sorted(gain_settings.items()):
                if voltage_range <= last_voltage:
                    self.gain = gain
                    self.max_voltage = voltage_range
                    logger.info(f"Zoomed in: new gain is {self.gain}x")
                    break

    def _start_measurement_thread(self):
        """
        Start the voltage measurement process in a separate thread.
        """
        self.running = True
        self.measurement_thread = threading.Thread(
            target=self._do_measurement, daemon=True
        )
        self.measurement_thread.start()

    def _do_measurement(self):
        """
        Reads the voltage from the ADS1115 ADC and adds this measurement to self.measurements.
        """
        try:
            # Initialize I2C bus and ADS1115 ADC
            i2c = busio.I2C(board.SCL, board.SDA)
            ads = ADS.ADS1115(i2c, address=self.i2c_address)
            
            # Set PGA gain dynamically
            ads.gain = self.gain
            logger.info(f"ADS1115 initialized with dynamic gain setting: {self.gain}x")
            
            # Available voltage ranges based on gain
            gain_ranges = {
                2/3: 6.144,  # +/-6.144V
                1: 4.096,    # +/-4.096V
                2: 2.048,    # +/-2.048V
                4: 1.024,    # +/-1.024V
                8: 0.512,    # +/-0.512V
                16: 0.256    # +/-0.256V
            }
            voltage_range = gain_ranges.get(self.gain, "Unknown")
            logger.info(f"Voltage measurement range: +/- {voltage_range}V")

            # Create analog input channel on the ADC
            chan = AnalogIn(ads, getattr(ADS, f'P{self.adc_channel}'))

            count = 0

            while self.running:
                try:
                    # Read voltage from ADC (already amplified by PGA hardware)
                    voltage = chan.voltage
                    
                    # Append the measurement with timestamp
                    self.measurements.append({
                        "voltage": voltage,
                        "time_ns": time.time_ns(),
                        "time": time.time()
                    })

                    # Adjust gain based on the latest measurement
                    self._adjust_gain(voltage)

                    count += 1

                    # Make sure we have enough samples
                    if count >= self.sample_size and len(self.measurements) < self.sample_size:
                        logger.warning(
                            f"Only {len(self.measurements)} readings "
                            f"for voltage measurement, please decrease measurement_interval"
                        )

                except Exception as e:
                    logger.error(f"Error during voltage measurement: {e}")

                time.sleep(self.measurement_interval)

        except Exception as e:
            logger.error(f"Failed to initialize ADS1115: {e}")

    def stop(self):
        """
        Stop the measurement process.
        """
        self.running = False
        if hasattr(self, 'measurement_thread'):
            self.measurement_thread.join()

    def read_voltage(self):
        """
        Calculate average voltage based on the measurements.
        """
        if not self.measurements:
            return 0.0

        # Return the average voltage from the measurements
        return sum(m["voltage"] for m in self.measurements) / len(self.measurements)


# Define Pydantic models for request/response validation
class VoltageSensorSettings(BaseModel):
    i2c_address: int = Field(
        0x48, description="I2C address of the ADS1115 (default 0x48 or 72 decimal)"
    )
    adc_channel: int = Field(
        0, description="ADC channel to read from (0-3)"
    )
    measurement_window: int = Field(
        10, description="Window size for measurements (in samples)"
    )
    measurement_interval: float = Field(
        0.1, description="Interval between measurements (in seconds)"
    )
    sample_size: int = Field(
        10, description="Number of samples needed for reliable measurement"
    )

    @validator("i2c_address")
    def validate_i2c_address(cls, v):
        if not (0x48 <= v <= 0x4B):
            raise ValueError("i2c_address must be between 0x48 and 0x4B (72-75)")
        return v

    @validator("adc_channel")
    def validate_adc_channel(cls, v):
        if not (0 <= v <= 3):
            raise ValueError("adc_channel must be between 0 and 3")
        return v

    @validator("measurement_window")
    def validate_window(cls, v):
        if v <= 0:
            raise ValueError("measurement_window must be positive")
        return v

    @validator("measurement_interval")
    def validate_interval(cls, v):
        if v <= 0:
            raise ValueError("measurement_interval must be positive")
        return v

    @validator("sample_size")
    def validate_sample_size(cls, v):
        if v <= 0:
            raise ValueError("sample_size must be positive")
        return v
