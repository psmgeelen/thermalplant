import time
import threading
from collections import deque
import gpiozero
from gpiozero.pins.lgpio import LGPIOFactory
from pydantic import BaseModel, Field, validator
from .utils import get_logger

logger = get_logger(__name__)

# Force gpiozero to use RPi.GPIO as the pin factory
gpiozero.Device.pin_factory = LGPIOFactory()

class RPMSensor(object):

    def __init__(
            self,
            gpio_pin: int,
            measurement_interval: int,
            measurement_window: int,
            sample_size: int,
    ):
        self.gpiopin = gpio_pin
        self.measurement_interval = measurement_interval
        self.measurement_window = measurement_window
        self.sample_size = sample_size
        self.measurements = deque(maxlen=self.measurement_window)
        self.running = False  # Flag to control the measurement thread
        self._start_measurement_thread()
        self.prior_state_count = 0

    def _start_measurement_thread(self):
        """
        Start the measurement process in a separate thread.
        """
        self.running = True
        self.measurement_thread = threading.Thread(
            target=self._do_measurement, daemon=True
        )
        self.measurement_thread.start()

    def _do_measurement(self):
        """
        Reads the state of the GPIO pin and adds this measurement to self.measurements.
        """
        prior_state = None
        pin = gpiozero.InputDevice(self.gpiopin)
        while self.running:  # Check self.running to allow clean shutdown
            try:
                # Read the state of the GPIO pin
                state = pin.value  # Returns 1 if pin is HIGH, 0 if LOW

                # Append the measurement to the measurements deque
                if prior_state is not None and prior_state != state:
                    self.measurements.append(
                        {"state": state, "time_ns": time.time_ns(), "time": time.time()}
                    )
                    if prior_state_count < self.sample_size:
                        logger.warning(
                            f"Only {prior_state_count} readings "
                            f"for measurement, please increase measurement_interval"
                        )
                    self.prior_state_count = 0

                if prior_state is state:
                    self.prior_state_count += 1

                prior_state = state

            except gpiozero.GPIOZeroError as gpe:
                logger.error(f"GPIO error during measurement: {gpe}")
            except ValueError as ve:
                logger.error(f"Value error during measurement: {ve}")
            except Exception as e:
                logger.error(f"Error during measurement: {e}")

            time.sleep(self.measurement_interval)

    def stop(self):
        """
        Stop the measurement process.
        """
        self.running = False
        self.measurement_thread.join()

    def read_rpm(self):
        """
        Calculate RPM based on the measurements.
        """
        # Implementation will depend on your specific RPM calculation logic
        if not self.measurements:
            return 0

        if len(self.measurements) < self.measurement_window:
            return 0

        timens = [m["time_ns"] for m in self.measurements]

        if self.prior_state_count > 100:
            # if the prior state hasn't changed for 100 measurements, we assume its not changing at all, rpms => 0
            rpm = 0
        else:
            # One revolution means 01-01-01-01 because we have 4 blades that we detect or not.
            # One revolution there is 8 entries. So the time for 1 revolution is the first and
            # last item, devided by length of the list (e.g. 200 items) multiplied by 8 entries
            # so if i have a list of 200 items, I have 25 revolutions.
            # The time difference between first and last item therefore needs to be devided
            # 25 revolutions
            revolution_time_ns = (timens[-1] - timens[0]) / (len(self.measurements) / 8)
            rpm = (60 * 1e9) / revolution_time_ns
        return rpm

# Define Pydantic models for request/response validation
class RPMSensorSettings(BaseModel):
    measurement_window: int = Field(
        100, description="Window size for measurements (in samples)"
    )
    measurement_interval: float = Field(
        0.001, description="Interval between measurements (in seconds)"
    )
    sample_size: int = Field(
        8, description="Number of samples needed to ensure not skipping a cycle"
    )

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
