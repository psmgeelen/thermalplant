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
        # Watchdog variables
        self.last_sample_time = 0
        self.max_silence_duration = 5.0  # seconds without data before watchdog triggers
        self.watchdog_thread = None
        self.watchdog_running = False

    def start(self, sample_duration: float = 5.0):
        if self.running:
            return

        self.running = True
        self.pyaudio_instance = pyaudio.PyAudio()
        self.chunks_per_sample = int((self.rate * sample_duration) / self.chunk_size)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        
        # Start watchdog
        self.watchdog_running = True
        self.last_sample_time = time.time()
        self.watchdog_thread = threading.Thread(target=self._watchdog_monitor, daemon=True)
        self.watchdog_thread.start()
        
    def _watchdog_monitor(self):
        """Monitors audio collection and automatically recovers if it stops"""
        logger.info("Audio collection watchdog started")
        while self.watchdog_running and self.running:
            try:
                time_since_last_sample = time.time() - self.last_sample_time
                if time_since_last_sample > self.max_silence_duration:
                    logger.warning(f"Audio collection appears to have stopped! "
                                  f"No samples for {time_since_last_sample:.1f} seconds. "
                                  f"Attempting recovery...")
                    
                    # Try to restart the stream
                    if self.stream:
                        try:
                            self.stream.stop_stream()
                            self.stream.close()
                        except Exception as e:
                            logger.error(f"Error closing stalled stream: {str(e)}")
                    
                    try:
                        # Reopen stream
                        self.stream = self.pyaudio_instance.open(
                            format=pyaudio.paInt16,
                            channels=self.channels,
                            rate=self.rate,
                            input=True,
                            input_device_index=self.device_index,
                            frames_per_buffer=self.chunk_size,
                        )
                        logger.info("Audio stream successfully restarted by watchdog")
                        self.last_sample_time = time.time()  # Reset timer after recovery
                    except Exception as e:
                        logger.error(f"Watchdog failed to restart audio stream: {str(e)}")
                        
                        # Try to completely reinitialize PyAudio
                        try:
                            if self.pyaudio_instance:
                                self.pyaudio_instance.terminate()
                            
                            # Create new PyAudio instance
                            self.pyaudio_instance = pyaudio.PyAudio()
                            
                            # Reopen stream with new instance
                            self.stream = self.pyaudio_instance.open(
                                format=pyaudio.paInt16,
                                channels=self.channels,
                                rate=self.rate,
                                input=True,
                                input_device_index=self.device_index,
                                frames_per_buffer=self.chunk_size,
                            )
                            logger.info("PyAudio completely reinitialized by watchdog")
                            self.last_sample_time = time.time()
                        except Exception as reinit_error:
                            logger.error(f"Watchdog failed to reinitialize PyAudio: {str(reinit_error)}")
            except Exception as e:
                logger.error(f"Error in audio watchdog: {str(e)}")
                
            # Check every second
            time.sleep(1.0)

    def _run(self):
        max_retries = 5
        retry_delay = 1.0
        retry_count = 0
        
        while self.running:
            try:
                # Initialize the audio stream
                logger.info(f"Initializing audio stream (attempt {retry_count + 1})")
                self.stream = self.pyaudio_instance.open(
                    format=pyaudio.paInt16,
                    channels=self.channels,
                    rate=self.rate,
                    input=True,
                    input_device_index=self.device_index,
                    frames_per_buffer=self.chunk_size,
                )
    
                # Reset retry count after successful initialization
                retry_count = 0
                
                # Update last sample time
                self.last_sample_time = time.time()
                
                # Collect initial sample
                if not self._collect_one_sample():
                    logger.warning("Failed to collect initial audio sample, retrying...")
                    continue
    
                # Main recording loop
                consecutive_failures = 0
                while self.running:
                    try:
                        if self._collect_one_sample():
                            # Reset failure counter on success
                            consecutive_failures = 0
                        else:
                            consecutive_failures += 1
                            logger.warning(f"Failed to collect audio sample ({consecutive_failures} consecutive failures)")
                            
                            # If too many consecutive failures, restart the stream
                            if consecutive_failures >= 3:
                                logger.warning("Too many consecutive collection failures, restarting stream")
                                break
                                
                        time.sleep(0.01)  # Prevent CPU overuse
                    except Exception as e:
                        logger.error(f"Error during audio collection: {str(e)}")
                        consecutive_failures += 1
                        
                        # Break loop to restart stream after too many errors
                        if consecutive_failures >= 3:
                            logger.warning("Too many consecutive errors, restarting stream")
                            break
                            
                        # Brief pause after error
                        time.sleep(0.5)
                
                # If we reach here, the inner loop broke - clean up the stream
                if self.stream:
                    try:
                        self.stream.stop_stream()
                        self.stream.close()
                        self.stream = None
                    except Exception as close_error:
                        logger.error(f"Error closing stream: {str(close_error)}")
                        self.stream = None
                
            except Exception as outer_error:
                logger.error(f"Critical error in recording loop: {str(outer_error)}")
                retry_count += 1
                
                # If too many retries, give up
                if retry_count >= max_retries and self.running:
                    logger.error(f"Failed to initialize audio after {max_retries} attempts, giving up")
                    self.running = False
                    break
                
                # Wait before retrying
                time.sleep(retry_delay)
                
                # Increase retry delay (exponential backoff)
                retry_delay = min(retry_delay * 1.5, 10.0)
                
                # Release existing resources before retry
                self._cleanup()
                
                # Try to reinitialize PyAudio on major failures
                if retry_count > 1 and self.pyaudio_instance:
                    try:
                        self.pyaudio_instance.terminate()
                        time.sleep(1.0)  # Give system time to release audio resources
                        self.pyaudio_instance = pyaudio.PyAudio()
                        logger.info("PyAudio reinitialized after failure")
                    except Exception as pa_error:
                        logger.error(f"Failed to reinitialize PyAudio: {str(pa_error)}")
        
        # Final cleanup when loop ends
        self._cleanup()

    def _collect_one_sample(self):
        frames = []
        for _ in range(self.chunks_per_sample):
            if not self.running:
                break
            try:
                data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                frames.append(data)
            except (IOError, OSError) as e:
                # Handle common audio stream errors
                logger.error(f"Error reading from audio stream: {str(e)}")
                # Break the loop to trigger recovery
                break
    
        if frames and self.running:
            audio_data = b"".join(frames)
            logger.info(f"Adding audio chunk..")
            self.audio_buffer.append(audio_data)
            self.sample_ready.set()
            self.sample_ready.clear()
            # Update last sample time for watchdog
            self.last_sample_time = time.time()
            return True
        return False

    def _cleanup(self):#
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
        self.watchdog_running = False
        
        # Stop main thread
        if self.thread:
            self.thread.join(timeout=2)
            self.thread = None
            
        # Stop watchdog thread
        if self.watchdog_thread:
            self.watchdog_thread.join(timeout=2)
            self.watchdog_thread = None

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
                        count = 0
                        for band_label, indices in self.spectrum_bands.items():
                            if indices:  # Only process bands with indices
                                # Average all frequencies in this band
                                band_values = [
                                    spectrum_avg[i]
                                    for i in indices
                                    if i < len(spectrum_avg)
                                ]
                                if band_values:
                                    incremented_band_label = f"spectrum_{count}_{band_label.split("spectrum_")[-1]}"
                                    spectrum_dict[incremented_band_label] = float(
                                        np.mean(band_values)
                                    )
                                    count += 1

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
        # Track number of successive failures to determine when to attempt recovery
        failure_count = 0
        last_successful_sync = time.time()
        recovery_in_progress = False
        
        while self.running:
            try:
                # Check health of processing loop
                if not hasattr(self, 'processing_loop') or not self.processing_loop:
                    logger.error("Processing loop unavailable, attempting recovery")
                    self._recover_audio_components()
                    time.sleep(1.0)
                    continue
                
                # Check health of recording loop 
                if not hasattr(self, 'recording_loop') or not self.recording_loop:
                    logger.error("Recording loop unavailable, attempting recovery")
                    self._recover_audio_components()
                    time.sleep(1.0)
                    continue
                
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
                    
                    # Reset failure tracking on successful sync
                    failure_count = 0
                    last_successful_sync = time.time()
                    recovery_in_progress = False
                else:
                    # Check if we've gone too long without successful data
                    time_since_last_sync = time.time() - last_successful_sync
                    if time_since_last_sync > 10.0 and not recovery_in_progress:  # 10 seconds without data
                        logger.warning(f"No audio data processed for {time_since_last_sync:.1f} seconds, attempting recovery")
                        self._recover_audio_components()
                        recovery_in_progress = True
                        time.sleep(1.0)
                        continue
                
                time.sleep(0.1)  # Prevent CPU overuse
            except AttributeError as ae:
                logger.error(f"Attribute error in sync thread: {str(ae)}")
                failure_count += 1
                if failure_count >= 3 and not recovery_in_progress:
                    logger.warning("Multiple attribute errors, attempting to recover audio components")
                    self._recover_audio_components()
                    recovery_in_progress = True
                    failure_count = 0
                time.sleep(0.5)
            except RuntimeError as re:
                logger.error(f"Runtime error in sync thread: {str(re)}")
                failure_count += 1
                if failure_count >= 3 and not recovery_in_progress:
                    logger.warning("Multiple runtime errors, attempting to recover audio components")
                    self._recover_audio_components()
                    recovery_in_progress = True
                    failure_count = 0
                time.sleep(0.5)
            except Exception as e:
                logger.error(f"Unexpected error in sync thread: {str(e)}")
                failure_count += 1
                if failure_count >= 3 and not recovery_in_progress:
                    logger.warning("Multiple errors in sync thread, attempting to recover audio components")
                    self._recover_audio_components()
                    recovery_in_progress = True  
                    failure_count = 0
                time.sleep(0.5)

    def read_mfcc(self):
        # If buffer is empty, check if we need recovery
        last_error = None
        max_retries = 2
        
        for attempt in range(max_retries):
            try:
                if not self.mfcc_buffer:
                    # Wait briefly for data to become available
                    if not self.data_ready_event.wait(timeout=2.0):
                        logger.warning(f"No MFCC data available after timeout (attempt {attempt+1}/{max_retries})")
                        if attempt == max_retries - 1:
                            raise RuntimeError("No MFCC data available after timeout")
                        else:
                            # Try recovery before final attempt
                            self._recover_audio_components()
                            time.sleep(1.0)
                            continue
    
                # Check that we have data
                if not self.mfcc_buffer:
                    logger.warning(f"No MFCC data in buffer (attempt {attempt+1}/{max_retries})")
                    if attempt == max_retries - 1:
                        raise RuntimeError("No MFCC data available")
                    else:
                        # Try recovery before final attempt
                        self._recover_audio_components()
                        time.sleep(1.0)
                        continue
                        
                return self.mfcc_buffer[-1]
                
            except Exception as e:
                last_error = e
                logger.error(f"Error in read_mfcc (attempt {attempt+1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    # Try recovery before final attempt
                    self._recover_audio_components()
                    time.sleep(1.0)
        
        # If we get here, all attempts failed
        logger.error("All attempts to read MFCC data failed")
        raise last_error or RuntimeError("Failed to read MFCC data after multiple attempts")
    
    def read_spectrum(self):
        # If buffer is empty, check if we need recovery
        last_error = None
        max_retries = 2
        
        for attempt in range(max_retries):
            try:
                if not self.spectrum_buffer:
                    # Wait briefly for data to become available
                    if not self.data_ready_event.wait(timeout=2.0):
                        logger.warning(f"No spectrum data available after timeout (attempt {attempt+1}/{max_retries})")
                        if attempt == max_retries - 1:
                            raise RuntimeError("No spectrum data available after timeout")
                        else:
                            # Try recovery before final attempt
                            self._recover_audio_components()
                            time.sleep(1.0)
                            continue
    
                # Check that we have data
                if not self.spectrum_buffer:
                    logger.warning(f"No spectrum data in buffer (attempt {attempt+1}/{max_retries})")
                    if attempt == max_retries - 1:
                        raise RuntimeError("No spectrum data available")
                    else:
                        # Try recovery before final attempt
                        self._recover_audio_components()
                        time.sleep(1.0)
                        continue
                            
                return self.spectrum_buffer[-1]
                
            except Exception as e:
                last_error = e
                logger.error(f"Error in read_spectrum (attempt {attempt+1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    # Try recovery before final attempt
                    self._recover_audio_components()
                    time.sleep(1.0)
        
        # If we get here, all attempts failed
        logger.error("All attempts to read spectrum data failed")
        raise last_error or RuntimeError("Failed to read spectrum data after multiple attempts")

    def read_all_audio(self):
        """Return a combined dictionary with all audio data"""
        result = {}
        
        # Try to get MFCC data
        try:
            mfcc_data = self.read_mfcc()
            result["mfcc"] = mfcc_data
        except Exception as e:
            logger.error(f"Error reading MFCC data: {str(e)}")
            result["mfcc"] = {"error": str(e)}
            
            # Try recovery once
            try:
                self._recover_audio_components()
                time.sleep(1.0)
                # Try again after recovery
                mfcc_data = self.read_mfcc()
                result["mfcc"] = mfcc_data
            except Exception as recovery_error:
                logger.error(f"Failed to get MFCC data after recovery: {str(recovery_error)}")
        
        # Try to get spectrum data
        try:
            spectrum_data = self.read_spectrum()
            result["spectrum"] = spectrum_data
        except Exception as e:
            logger.error(f"Error reading spectrum data: {str(e)}")
            result["spectrum"] = {"error": str(e)}
            
            # Only try recovery if we haven't already tried above
            if "error" not in result.get("mfcc", {}):
                try:
                    self._recover_audio_components()
                    time.sleep(1.0)
                    # Try again after recovery
                    spectrum_data = self.read_spectrum()
                    result["spectrum"] = spectrum_data
                except Exception as recovery_error:
                    logger.error(f"Failed to get spectrum data after recovery: {str(recovery_error)}")
    
        return result

    def _recover_audio_components(self):
        """Internal method to recover audio components without fully restarting"""
        logger.info("Attempting to recover audio components")
        
        # First, try to stop existing components
        if hasattr(self, "processing_loop"):
            try:
                self.processing_loop.stop()
            except Exception as e:
                logger.warning(f"Error stopping processing loop during recovery: {str(e)}")
        
        if hasattr(self, "recording_loop"):
            try:
                self.recording_loop.stop()
            except Exception as e:
                logger.warning(f"Error stopping recording loop during recovery: {str(e)}")
        
        # Brief pause to allow resources to be released
        time.sleep(1.0)
        
        try:
            # Recreate recording loop
            matches = self._find_audio_device()
            if len(matches) == 0:
                logger.error("No audio device found during recovery")
                return
            
            # Reinitialize recording loop
            self.recording_loop = RecordingLoop(
                device_index=self.device_index,
                rate=self.rate,
                channels=self.channels,
                chunk_size=1024,
            )
            self.recording_loop.start(sample_duration=self.sample_duration)
            
            # Wait for first audio sample
            if not self.recording_loop.wait_for_data(timeout=5):
                logger.error("Could not record audio during recovery")
                return
                
            # Recreate processing loop
            self.processing_loop = ProcessingLoop(
                rate=self.rate, mfcc_count=self.mfcc_count, n_fft=2048
            )
            self.processing_loop.start(self.recording_loop.get_audio_data)
            
            # Wait for initial processing
            if not self.processing_loop.wait_for_processing(timeout=5):
                logger.error("Could not process audio during recovery")
                return
                
            # Get initial processed data to validate recovery
            initial_mfcc = self.processing_loop.get_latest_mfcc()
            initial_spectrum = self.processing_loop.get_latest_spectrum()
            
            if initial_mfcc is not None and initial_spectrum is not None:
                logger.info("Audio components successfully recovered")
                self.mfcc_buffer.append(initial_mfcc)
                self.spectrum_buffer.append(initial_spectrum)
                self.data_ready_event.set()
            else:
                logger.error("Audio recovery failed: no data available after reinitializing components")
                
        except Exception as e:
            logger.error(f"Error during audio component recovery: {str(e)}")
    
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
