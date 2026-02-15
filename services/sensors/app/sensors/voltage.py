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

class VoltageSensor:

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

        self.gain = 2 / 3
        self.max_voltage = 6.144
        self.measurements = deque(maxlen=self.measurement_window)
        self.running = False
        self._lock = threading.Lock()

        # Initialize I2C and ADS1115 once
        i2c = busio.I2C(board.SCL, board.SDA)
        self.ads = ADS.ADS1115(i2c, address=self.i2c_address)
        self.ads.gain = self.gain
        self.chan = AnalogIn(self.ads, getattr(ADS, f'P{self.adc_channel}'))

        logger.info(f"ADS1115 initialized on 0x{self.i2c_address:X} with gain {self.gain}x")
        self._start_measurement_thread()

    def _adjust_gain(self, last_voltage):
        gain_settings = {
            2/3: 6.144,
            1: 4.096,
            2: 2.048,
            4: 1.024,
            8: 0.512,
            16: 0.256
        }

        threshold = 0.8 * self.max_voltage
        new_gain = self.gain

        if last_voltage > threshold:
            for gain, v_range in sorted(gain_settings.items(), reverse=True):
                if v_range >= last_voltage:
                    new_gain = gain
                    break
        elif last_voltage < threshold:
            for gain, v_range in sorted(gain_settings.items()):
                if v_range <= last_voltage:
                    new_gain = gain
                    break

        if new_gain != self.gain:
            self.gain = new_gain
            self.max_voltage = gain_settings[new_gain]
            self.ads.gain = self.gain
            self.chan = AnalogIn(self.ads, getattr(ADS, f'P{self.adc_channel}'))
            logger.info(f"Adjusted gain to {self.gain}x, range +/- {self.max_voltage}V")

    def _start_measurement_thread(self):
        self.running = True
        self.measurement_thread = threading.Thread(
            target=self._do_measurement, daemon=True
        )
        self.measurement_thread.start()

    def _do_measurement(self):
        try:
            count = 0

            while self.running:
                try:
                    with self._lock:
                        voltage = self.chan.voltage

                    self.measurements.append({
                        "voltage": voltage,
                        "time_ns": time.time_ns(),
                        "time": time.time()
                    })

                    self._adjust_gain(voltage)

                    count += 1
                    if count >= self.sample_size and len(self.measurements) < self.sample_size:
                        logger.warning(
                            f"Only {len(self.measurements)} readings so far, consider reducing measurement_interval"
                        )

                except Exception as e:
                    logger.error(f"Error during voltage measurement: {e}")

                time.sleep(self.measurement_interval)

        except Exception as e:
            logger.error(f"Failed to start measurement thread: {e}")

    def stop(self):
        self.running = False
        if hasattr(self, 'measurement_thread'):
            self.measurement_thread.join()

    def read_voltage(self):
        with self._lock:
            if not self.measurements:
                return 0.0
            return sum(m["voltage"] for m in self.measurements) / len(self.measurements)


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
        0.5, description="Interval between measurements (in seconds)"
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
