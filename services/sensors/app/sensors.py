import spidev
from collections import deque
import time
import gpiozero
import logging
from statistics import mean
import threading
import asyncio
import numpy as np
import pyaudio
import librosa
import os
from gpiozero.pins.lgpio import LGPIOFactory


logger = logging.getLogger("sensors")
# Force gpiozero to use RPi.GPIO as the pin factory
gpiozero.Device.pin_factory = LGPIOFactory()


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
            raise ValueError(f"SPI port {spi_port} is not available. Check your system configuration.")

        self.spi.max_speed_hz = max_speed_hz
        self.spi.mode = mode

        print(f"SPI initialized on port {spi_port}, chip select {chip_select}, speed {max_speed_hz}Hz, mode {mode}")

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
            print(f"Temperature reading error: {str(e)}")
            raise

    def _convert_to_temperature(self, raw_data):
        """
        Convert raw sensor data to temperature in Celsius.
        """
        # Combine the two bytes into a 16-bit number
        combined_data = (raw_data[0] << 8) | raw_data[1]

        # Check for an error signal from the sensor (bit 2 set indicates an error)
        if combined_data & 0x4:
            return float('NaN')

        # Extract the 12-bit temperature data (ignoring the 3 least significant bits)
        temp_data = combined_data >> 3

        # Convert the temperature data to Celsius
        return temp_data * self.CELSIUS_PER_BIT

    def close(self):
        """Close the SPI connection."""
        self.spi.close()
        print("SPI connection closed.")

class RPMSensor(object):

    def __init__(self, gpio_pin: int, measurement_interval: int, measurement_window: int, sample_size: int):
        self.gpiopin = gpio_pin
        self.measurement_interval = measurement_interval
        self.measurement_window = measurement_window
        self.sample_size = sample_size
        self.measurements = deque(maxlen=self.measurement_window)
        self.running = False  # Flag to control the measurement thread
        self._start_measurement_thread()

    def _start_measurement_thread(self):
        """
        Start the measurement process in a separate thread.
        """
        self.running = True
        self.measurement_thread = threading.Thread(target=self._do_measurement, daemon=True)
        self.measurement_thread.start()

    def _do_measurement(self):
        """
        Reads the state of the GPIO pin and adds this measurement to self.measurements.
        """
        prior_state = None
        prior_state_count = 0
        pin = gpiozero.InputDevice(self.gpiopin)
        while True:
            try:

                # Read the state of the GPIO pin
                state = pin.value  # Returns 1 if pin is HIGH, 0 if LOW

                # Append the measurement to the measurements deque
                if prior_state is not None and prior_state != state:
                    self.measurements.append({
                        "state": state,
                        "time_ns": time.time_ns(),
                        "time": time.time()
                    })
                    if prior_state_count < self.sample_size:
                        logger.warning(f"Only {prior_state_count} readings for measurement, please increase measurement_interval")
                    prior_state_count = 0

                if prior_state is state:
                    prior_state_count += 1

                prior_state = state

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

        # One revolution means 01-01-01-01 because we have 4 blades that we detect or not. One revolution there is 8 entries
        # So the time for 1 revolution is the first and last item, devided by length of the list (e.g. 200 items) multiplied by 8 entries
        # so if i have a list of 200 items, I have 25 revolutions. The time difference between first and last item therefore needs to be devided by 25 revolutions
        revolution_time_ns = (timens[-1] - timens[0])/(len(self.measurements)/8)
        rpm = (60 * 1e9) / revolution_time_ns
        return rpm

class AudioSensor(object):
    """
    A microphone audio sensor that records audio in 5-second increments and processes
    it asynchronously to extract MFCC features.
    """

    def __init__(self, rate=40000, channels=1, sample_duration=5, mfcc_count=13, buffer_size=10):
        """
        Initialize the audio sensor with specified parameters.
        rate: Sampling rate for the microphone.
        channels: Number of audio channels (1 for mono).
        sample_duration: Duration of each recording in seconds.
        mfcc_count: Number of MFCC components to extract.
        buffer_size: The maximum number of recordings to store in the deque.
        """
        self.rate = rate
        self.channels = channels
        self.sample_duration = sample_duration
        self.mfcc_count = mfcc_count
        self.buffer_size = buffer_size

        self.audio_buffer = deque(maxlen=self.buffer_size)
        self.mfcc_buffer = deque(maxlen=1)

        self.running = False  # Flag to control the recording and processing threads

        # Initialize PyAudio
        self.p = pyaudio.PyAudio()

    def start(self):
        """
        Start the audio recording and MFCC processing in separate threads.
        """
        self.running = True
        self.recording_thread = threading.Thread(target=self._record_audio, daemon=True)
        self.recording_thread.start()

        asyncio.run(self._process_mfcc())  # Start processing MFCC asynchronously

    def stop(self):
        """
        Stop the audio recording and processing threads.
        """
        self.running = False
        self.recording_thread.join()

    def read_audio(self):
        """
        Retrieve the most recent MFCC features from the processing buffer.

        Returns:
            numpy.ndarray: The most recent MFCC features, or None if no data is available.
        """
        if len(self.mfcc_buffer) > 0:
            return self.mfcc_buffer.popleft()  # Retrieve the newest MFCC
        else:
            return None  # No MFCCs are currently available

    def _record_audio(self):
        """
        Continuously records audio in 5-second increments and appends the data to the buffer.
        """
        stream = self.p.open(format=pyaudio.paInt16,
                             channels=self.channels,
                             rate=self.rate,
                             input=True,
                             frames_per_buffer=self.rate * self.sample_duration)

        while self.running:
            audio_data = stream.read(self.rate * self.sample_duration)
            self.audio_buffer.append(audio_data)  # Store the recording in the deque
            time.sleep(self.sample_duration)  # Record every SAMPLE_DURATION seconds

    async def _process_mfcc(self):
        """
        Asynchronously processes audio data from the deque and extracts MFCC features.
        """
        while self.running:
            if len(self.audio_buffer) > 0:
                audio_data = self.audio_buffer.popleft()  # Get the oldest audio recording

                # Convert audio data to numpy array (from bytes)
                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)

                # Normalize audio data
                audio_np /= np.max(np.abs(audio_np))  # Normalize to range [-1, 1]

                # Extract MFCC features using librosa
                mfcc = librosa.feature.mfcc(y=audio_np, sr=self.rate, n_mfcc=self.mfcc_count)

                # Log the shape of the MFCC array
                logger.info(f"MFCC components: {mfcc.shape}")
                self.mfcc_buffer.append(mfcc)

                # Add a small delay to simulate processing time
                await asyncio.sleep(0.1)
            else:
                await asyncio.sleep(0.1)

    def close(self):
        """Close the PyAudio stream."""
        self.p.terminate()
        logger.info("PyAudio connection closed.")

