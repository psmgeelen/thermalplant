import spidev
from collections import deque
import time
import gpiozero
import logging
import threading
from typing import Optional
import numpy as np
import pyaudio
import librosa
from gpiozero.pins.lgpio import LGPIOFactory
from pydantic import BaseModel, Field, validator

logger = logging.getLogger("sensors-backend")
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
        prior_state_count = 0
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
                    prior_state_count = 0
    
                if prior_state is state:
                    prior_state_count += 1
    
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

        # One revolution means 01-01-01-01 because we have 4 blades that we detect or not.
        # One revolution there is 8 entries. So the time for 1 revolution is the first and
        # last item, devided by length of the list (e.g. 200 items) multiplied by 8 entries
        # so if i have a list of 200 items, I have 25 revolutions.
        # The time difference between first and last item therefore needs to be devided
        # 25 revolutions
        revolution_time_ns = (timens[-1] - timens[0]) / (len(self.measurements) / 8)
        rpm = (60 * 1e9) / revolution_time_ns
        return rpm


class RecordingLoop:
    def __init__(
        self, device_index: int, rate: int, channels: int, chunk_size: int = 1024
    ):
        self.device_index = device_index
        self.rate = rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.running = False
        self.thread = None
        self.audio_buffer = deque(maxlen=10)
        self.pyaudio_instance = None
        self.stream = None
        self.sample_ready = threading.Event()

    def start(self, sample_duration: float = 5.0):
        if self.running:
            return

        self.running = True
        self.pyaudio_instance = pyaudio.PyAudio()
        self.chunks_per_sample = int((self.rate * sample_duration) / self.chunk_size)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        try:
            self.stream = self.pyaudio_instance.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self.chunk_size,
            )

            # Collect initial sample
            self._collect_one_sample()

            # Main recording loop
            while self.running:
                try:
                    self._collect_one_sample()
                    time.sleep(0.01)  # Prevent CPU overuse
                except Exception as e:
                    logger.error(f"Error during audio collection: {str(e)}")
                    if self.stream:
                        try:
                            self.stream.stop_stream()
                            self.stream.close()
    
                            # Reopen stream
                            self.stream = self.pyaudio_instance.open(
                                format=pyaudio.paInt16,
                                channels=self.channels,
                                rate=self.rate,
                                input=True,
                                input_device_index=self.device_index,
                                frames_per_buffer=self.chunk_size,
                            )
                        except Exception as reopen_error:
                            # If reopening fails, break the loop to prevent orphaned processes
                            logger.error(f"Failed to reopen audio stream: {str(reopen_error)}")
                            self.running = False
                            break
        except Exception as outer_error:
            logger.error(f"Critical error in recording loop: {str(outer_error)}")
            self.running = False
        finally:
            self._cleanup()

    def _collect_one_sample(self):
        frames = []
        for _ in range(self.chunks_per_sample):
            if not self.running:
                break
            data = self.stream.read(self.chunk_size, exception_on_overflow=False)
            frames.append(data)

        if frames and self.running:
            audio_data = b"".join(frames)
            self.audio_buffer.append(audio_data)
            self.sample_ready.set()
            self.sample_ready.clear()

    def _cleanup(self):
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None
            except Exception as e:
                logger.warning(f"Error while cleaning up audio stream: {str(e)}")

    def stop(self):
        if not self.running:
            return

        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
            self.thread = None

        self._cleanup()

        if self.pyaudio_instance:
            self.pyaudio_instance.terminate()
            self.pyaudio_instance = None

    def get_audio_data(self) -> Optional[bytes]:
        if self.audio_buffer:
            return self.audio_buffer.popleft()
        return None

    def has_data(self) -> bool:
        return len(self.audio_buffer) > 0

    def wait_for_data(self, timeout=None):
        return self.sample_ready.wait(timeout=timeout)


class ProcessingLoop:
    def __init__(self, rate: int, mfcc_count: int = 50, n_fft: int = 2048):
        self.rate = rate
        self.mfcc_count = mfcc_count
        self.n_fft = n_fft

        # Calculate frequency bins for spectrum
        self.freq_bins = librosa.fft_frequencies(sr=rate, n_fft=n_fft)

        # For MFCC, approximate the center frequencies (rough approximation)
        self.mfcc_freqs = librosa.mel_frequencies(
            n_mels=mfcc_count, fmin=0, fmax=rate / 2
        )

        self.running = False
        self.thread = None
        self.mfcc_buffer = deque(maxlen=3)
        self.spectrum_buffer = deque(maxlen=3)
        self.data_ready = False
        self.processing_event = threading.Event()

        # Setup frequencies only once
        self.mfcc_labels = [
            f"mfcc_{i+1}_{int(round(freq))}Hz" for i, freq in enumerate(self.mfcc_freqs)
        ]

        # Group spectrum frequencies into bands for easier interpretation
        self.spectrum_bands = {}
        band_size = 100  # Hz per band
        for i, freq in enumerate(self.freq_bins):
            if i >= 400:  # Limit to first 400 frequency bins for practicality
                break
            band = (int(round(freq)) // band_size) * band_size
            band_label = f"spectrum_{band}Hz_{band+band_size}Hz"
            if band_label not in self.spectrum_bands:
                self.spectrum_bands[band_label] = []
            self.spectrum_bands[band_label].append(i)

    def start(self, audio_getter):
        if self.running:
            return

        self.running = True
        self.audio_getter = audio_getter
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while self.running:
            try:
                # Pull pattern: check if there is data to process
                audio_data = self.audio_getter()

                if audio_data:
                    # Process the data
                    audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(
                        np.float32
                    )

                    # Process if non-empty
                    if len(audio_np) > 0:
                        # Normalize audio
                        max_abs = np.max(np.abs(audio_np))
                        if max_abs > 0:
                            audio_np /= max_abs

                        # Extract MFCC features
                        mfcc = librosa.feature.mfcc(
                            y=audio_np, sr=self.rate, n_mfcc=self.mfcc_count
                        )

                        # Store result as labeled dictionary
                        mfcc_average = np.mean(mfcc, axis=1)
                        mfcc_dict = {
                            label: float(value)
                            for label, value in zip(self.mfcc_labels, mfcc_average)
                        }
                        self.mfcc_buffer.append(mfcc_dict)

                        # Calculate spectrum (FFT magnitudes)
                        D = np.abs(librosa.stft(audio_np, n_fft=self.n_fft))
                        # Convert to power spectrum (squared magnitude)
                        spectrum = D**2

                        # Convert to decibels for better visualization
                        spectrum_db = librosa.power_to_db(spectrum, ref=np.max)

                        # Average across time frames
                        spectrum_avg = np.mean(spectrum_db, axis=1)

                        # Create labeled spectrum dictionary by bands
                        spectrum_dict = {}
                        for band_label, indices in self.spectrum_bands.items():
                            if indices:  # Only process bands with indices
                                # Average all frequencies in this band
                                band_values = [
                                    spectrum_avg[i]
                                    for i in indices
                                    if i < len(spectrum_avg)
                                ]
                                if band_values:
                                    spectrum_dict[band_label] = float(
                                        np.mean(band_values)
                                    )

                        self.spectrum_buffer.append(spectrum_dict)

                        self.data_ready = True
                        self.processing_event.set()
                else:
                    # No data available - wait a bit
                    time.sleep(0.5)
            except ValueError as ve:
                # Log specific value errors (common with audio processing)
                logger.error(f"Value error in audio processing: {str(ve)}")
                time.sleep(0.5)
            except np.linalg.LinAlgError as lae:
                # Linear algebra errors can occur during audio processing
                logger.error(f"Linear algebra error in audio processing: {str(lae)}")
                time.sleep(0.5)
            except TypeError as te:
                # Type errors might occur with unexpected audio data
                logger.error(f"Type error in audio processing: {str(te)}")
                time.sleep(0.5)
            except Exception as e:
                # Log the exception for debugging
                logger.error(f"Unexpected error in processing loop: {str(e)}")
                # If processing fails, make sure we don't create orphaned processes
                if not self.running:
                    break
                time.sleep(0.5)

    def stop(self):
        if not self.running:
            return

        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
            self.thread = None

    def get_latest_mfcc(self) -> Optional[dict]:
        """Return MFCC data as a dictionary with frequency labels"""
        if self.mfcc_buffer:
            return self.mfcc_buffer[-1]
        return None

    def get_latest_spectrum(self) -> Optional[dict]:
        """Return spectrum data as a dictionary with frequency band labels"""
        if self.spectrum_buffer:
            return self.spectrum_buffer[-1]
        return None

    def is_ready(self) -> bool:
        return self.data_ready

    def wait_for_processing(self, timeout=None):
        return self.processing_event.wait(timeout=timeout)


class AudioHandler:
    def __init__(
        self,
        rate: int = 44100,
        channels: int = 1,
        sample_duration: float = 5.0,
        mfcc_count: int = 50,
        spectrum_count: int = 1000,
        buffer_size: int = 3,
    ):
        self.rate = rate
        self.channels = channels
        self.sample_duration = sample_duration
        self.mfcc_count = mfcc_count
        self.buffer_size = buffer_size
        self.data_ready_event = threading.Event()
        self.running = True

        # Find audio device
        matches = self._find_audio_device()
        if len(matches) == 0:
            raise RuntimeError("No audio device available")
        self.device_index = matches[0]["index"]
        # Initialize components
        self._initialize_audio_components()

    def _find_audio_device(self, match_on: str = "USB Audio") -> list:
        matches = []
        p = pyaudio.PyAudio()

        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            logging.info(f"Found audio devices: {info}")
            if match_on in info["name"] and info["maxInputChannels"] == 1:
                matches.append(info)
                logging.info(f"Found a match!: {info}")
        return matches

    def _initialize_audio_components(self):
        # Create and start recording loop
        self.recording_loop = RecordingLoop(
            device_index=self.device_index,
            rate=self.rate,
            channels=self.channels,
            chunk_size=1024,
        )
        self.recording_loop.start(sample_duration=self.sample_duration)

        try:
            # Wait for first audio sample
            if not self.recording_loop.wait_for_data(timeout=10):
                self.recording_loop.stop()
                raise RuntimeError("Could not record initial audio sample")
        
            # Create and start processing loop
            self.processing_loop = ProcessingLoop(
                rate=self.rate, mfcc_count=self.mfcc_count, n_fft=2048
            )
            self.processing_loop.start(self.recording_loop.get_audio_data)
        
            # Initialize buffer for processed data
            self.mfcc_buffer = deque(maxlen=self.buffer_size)
            self.spectrum_buffer = deque(maxlen=self.buffer_size)
        
            # Wait for initial processing
            if not self.processing_loop.wait_for_processing(timeout=20):
                self.recording_loop.stop()
                self.processing_loop.stop()
                raise RuntimeError("Could not process initial audio sample")
        except Exception as e:
            # Clean up any initialized components before re-raising
            if hasattr(self, 'recording_loop'):
                try:
                    self.recording_loop.stop()
                except Exception as cleanup_error:
                    logger.error(f"Error stopping recording loop during "
                                 f"initialization failure: {cleanup_error}")
            if hasattr(self, 'processing_loop'):
                try:
                    self.processing_loop.stop()
                except Exception as cleanup_error:
                    logger.error(f"Error stopping processing loop during "
                                 f"initialization failure: {cleanup_error}")
            # Re-raise the original exception
            raise RuntimeError(f"Audio component initialization failed: {str(e)}") from e

        # Get initial processed data
        initial_mfcc = self.processing_loop.get_latest_mfcc()
        initial_spectrum = self.processing_loop.get_latest_spectrum()

        if initial_mfcc is not None:
            self.mfcc_buffer.append(initial_mfcc)
        else:
            self.recording_loop.stop()
            self.processing_loop.stop()
            raise RuntimeError("Failed to get initial MFCC data")

        if initial_spectrum is not None:
            self.spectrum_buffer.append(initial_spectrum)
        else:
            self.recording_loop.stop()
            self.processing_loop.stop()
            raise RuntimeError("Failed to get initial spectrum data")

        # Start background thread to sync processed data
        self.sync_thread = threading.Thread(
            target=self._sync_processed_data, daemon=True
        )
        self.sync_thread.start()

    def _sync_processed_data(self):
        while self.running:
            try:
                if self.processing_loop.is_ready():
                    # Sync MFCC data
                    latest_mfcc = self.processing_loop.get_latest_mfcc()
                    if latest_mfcc is not None:
                        # For dictionaries, we just replace them (no need for np.array_equal)
                        self.mfcc_buffer.append(latest_mfcc)

                    # Sync spectrum data
                    latest_spectrum = self.processing_loop.get_latest_spectrum()
                    if latest_spectrum is not None:
                        self.spectrum_buffer.append(latest_spectrum)

                    self.data_ready_event.set()
                time.sleep(0.1)  # Prevent CPU overuse
            except AttributeError as ae:
                logger.error(f"Attribute error in sync thread (likely during shutdown): {str(ae)}")
                self.running = False
                break
            except RuntimeError as re:
                logger.error(f"Runtime error in sync thread: {str(re)}")
                self.running = False
                break
            except Exception as e:
                logger.error(f"Unexpected error in sync thread: {str(e)}")
                # If sync fails, stop everything to prevent orphaned processes
                self.running = False
                break

    def read_mfcc(self):
        if not self.mfcc_buffer:
            # Wait briefly for data to become available
            if not self.data_ready_event.wait(timeout=2.0):
                raise RuntimeError("No MFCC data available after timeout")

        # Check that we have data
        if not self.mfcc_buffer:
            raise RuntimeError("No MFCC data available")

        return self.mfcc_buffer[-1]

    def read_spectrum(self):
        if not self.spectrum_buffer:
            # Wait briefly for data to become available
            if not self.data_ready_event.wait(timeout=2.0):
                raise RuntimeError("No spectrum data available after timeout")

        # Check that we have data
        if not self.spectrum_buffer:
            raise RuntimeError("No spectrum data available")

        return self.spectrum_buffer[-1]

    def read_all_audio(self):
        """Return a combined dictionary with all audio data"""
        mfcc_data = self.read_mfcc()
        spectrum_data = self.read_spectrum()

        return {"mfcc": mfcc_data, "spectrum": spectrum_data}

    def close(self):
        self.running = False

        # Stop all components
        if hasattr(self, "processing_loop"):
            try:
                self.processing_loop.stop()
            except Exception as e:
                logger.warning(f"Error stopping processing loop: {str(e)}")
    
        if hasattr(self, "recording_loop"):
            try:
                self.recording_loop.stop()
            except Exception as e:
                logger.warning(f"Error stopping recording loop: {str(e)}")
    
        # Wait for sync thread to finish
        if hasattr(self, "sync_thread") and self.sync_thread.is_alive():
            try:
                self.sync_thread.join(timeout=1)
            except Exception as e:
                logger.warning(f"Error joining sync thread: {str(e)}")


# Define Pydantic models for request/response validation
class RPMSensorSettings(BaseModel):
    measurement_window: int = Field(
        ..., description="Window size for measurements (in samples)"
    )
    measurement_interval: float = Field(
        ..., description="Interval between measurements (in seconds)"
    )
    sample_size: int = Field(
        ..., description="Number of samples needed to ensure not skipping a cycle"
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


class AudioHandlerSettings(BaseModel):
    sample_duration: float = Field(
        ..., description="Duration of audio samples in seconds"
    )
    mfcc_count: int = Field(..., description="Number of MFCC coefficients to extract")
    buffer_size: int = Field(
        ..., description="Size of the buffer for storing processed data"
    )

    @validator("sample_duration")
    def validate_duration(cls, v):
        if v <= 0:
            raise ValueError("sample_duration must be positive")
        return v

    @validator("mfcc_count")
    def validate_mfcc_count(cls, v):
        if v <= 0:
            raise ValueError("mfcc_count must be positive")
        if v > 128:
            raise ValueError("mfcc_count should be <= 128")
        return v

    @validator("buffer_size")
    def validate_buffer_size(cls, v):
        if v <= 0:
            raise ValueError("buffer_size must be positive")
        return v
