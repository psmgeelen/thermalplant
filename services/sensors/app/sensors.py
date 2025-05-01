import spidev
from collections import deque
import time
import gpiozero
import logging
import threading
from typing import Optional
from log_utils import get_logger  # Import the new queue-based logger utility
import numpy as np
import librosa
from gpiozero.pins.lgpio import LGPIOFactory
from pydantic import BaseModel, Field, validator
import pulsectl
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = get_logger("sensors-backend")
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


import asyncio
import pulsectl
import sys
import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor

class PipeWireRecordingLoop:
    def __init__(
        self, device_index: int, rate: int, channels: int, chunk_size: int = 1024
    ):
        self.device_index = device_index
        self.device_name = None  # Will be set when finding the device
        self.preferred_rate = rate  # Store the preferred rate
        self.rate = rate  # Will be adjusted if preferred rate isn't supported
        self.channels = channels
        self.chunk_size = chunk_size
        self.running = False
        self.thread = None
        self.audio_buffer = deque(maxlen=10)
        self.stream = None
        self.sample_ready = threading.Event()
        self.pulse = None  # Will hold PipeWire/PulseAudio client
        self.async_loop = None  # Will hold asyncio event loop
        
        # Watchdog variables
        self.last_sample_time = 0
        self.max_silence_duration = 5.0  # seconds without data before watchdog triggers
        self.watchdog_thread = None
        self.watchdog_running = False
        
        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Store device capabilities
        self.supported_rates = []
        self.default_rate = None

    def start(self, sample_duration: float = 5.0):
        if self.running:
            return

        self.running = True
        self.chunks_per_sample = int((self.rate * sample_duration) / self.chunk_size)
        
        # Create and start the main recording thread
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
                    
                    # Signal the main thread to restart
                    if self.async_loop and self.running:
                        try:
                            # Create a new client and try to recover
                            self._cleanup_pulse_client()
                            time.sleep(1.0)  # Brief pause before recovery
                            
                            # Reinitialize PipeWire
                            self._run_in_event_loop(self._setup_pipewire)
                            logger.info("PipeWire connection reinitialized by watchdog")
                            self.last_sample_time = time.time()  # Reset timer after recovery
                        except Exception as e:
                            logger.error(f"Watchdog failed to reinitialize PipeWire: {str(e)}")
            except Exception as e:
                logger.error(f"Error in audio watchdog: {str(e)}")
                
            # Check every second
            time.sleep(1.0)

    def _run(self):
        max_retries = 5
        retry_delay = 1.0
        retry_count = 0
        
        # Create an event loop for this thread
        self.async_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.async_loop)
        
        while self.running:
            try:
                # Initialize PipeWire connection
                logger.info(f"Initializing PipeWire connection (attempt {retry_count + 1})")
                self._run_in_event_loop(self._setup_pipewire)
                
                # Reset retry count after successful initialization
                retry_count = 0
                
                # Update last sample time
                self.last_sample_time = time.time()
                
                # Start the recording process
                self._run_in_event_loop(self._record_audio)
                
                # If we reach here, the recording has stopped - wait before retrying
                time.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Critical error in recording loop: {str(e)}")
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
                self._cleanup_pulse_client()
                
                # Give system time to release audio resources
                time.sleep(1.0)
        
        # Final cleanup when loop ends
        self._cleanup_pulse_client()
        
        # Close event loop
        if self.async_loop:
            self.async_loop.close()
            self.async_loop = None

    def _run_in_event_loop(self, func):
        """Helper to run async functions in the event loop"""
        if not self.async_loop:
            raise RuntimeError("Event loop is not initialized")
        
        future = asyncio.run_coroutine_threadsafe(func(), self.async_loop)
        return future.result()

    async def _setup_pipewire(self):
        """Initialize PipeWire connection and find appropriate source"""
        logger.info("########testing############")
        try:
            # Create a new pulse client
            self.pulse = pulsectl.Pulse('audio-recorder')
            
            # Get available sources
            sources = self.pulse.source_list()
            
            # Find appropriate source based on the device index or name
            source = None
            
            if self.device_name:
                # Try to find by name first if we have it from previous initialization
                for s in sources:
                    if self.device_name in s.name:
                        source = s
                        break
            
            # If not found by name, use the index
            if not source:
                if 0 <= self.device_index < len(sources):
                    source = sources[self.device_index]
                    self.device_name = source.name  # Remember the name for future recovery
                else:
                    # Fall back to default source if index is out of range
                    server_info = self.pulse.server_info()
                    default_source_name = server_info.default_source_name
                    for s in sources:
                        if s.name == default_source_name:
                            source = s
                            self.device_name = source.name
                            break
            
            if not source:
                # As a last resort, try to find any USB Audio source
                for s in sources:
                    if "USB Audio" in s.name:
                        source = s
                        self.device_name = source.name
                        break
                        
            if not source:
                # If still no source found, use the first available source
                if sources:
                    source = sources[0]
                    self.device_name = source.name
            
            if not source:
                raise RuntimeError("No suitable audio source found")
                
            # Determine the default sample rate and other capabilities
            self._detect_device_capabilities(source)
            
            # Adjust rate if needed
            self._adjust_sample_rate()
                
            logger.info(f"Selected audio source: {source.name} (using rate: {self.rate}Hz)")
            return source
            
        except Exception as e:
            logger.error(f"Failed to setup PipeWire: {str(e)}")
            self._cleanup_pulse_client()
            raise
            
    def _detect_device_capabilities(self, source):
        """Detect capabilities (sample rates, etc.) of the selected audio source"""
        try:
            # Get source information
            source_info = source
            
            # Store the sample rate from the source
            if hasattr(source_info, 'sample_spec') and hasattr(source_info.sample_spec, 'rate'):
                self.default_rate = source_info.sample_spec.rate
                logger.info(f"Detected default sample rate: {self.default_rate}Hz")
            else:
                # Fallback to common rates if we can't detect
                self.default_rate = 44100
                logger.info(f"Could not detect device rate, using fallback: {self.default_rate}Hz")
            
            # Common sample rates to test (could be expanded)
            test_rates = [8000, 11025, 16000, 22050, 32000, 44100, 48000, 96000]
            
            # Start with default and common rates
            self.supported_rates = [self.default_rate]
            if self.default_rate not in test_rates:
                test_rates.append(self.default_rate)
            
            # Log device capabilities
            logger.info(f"Audio device capabilities for {source_info.name}:")
            logger.info(f"  Default rate: {self.default_rate}Hz")
            logger.info(f"  Channels: {source_info.sample_spec.channels}")
            
            # Note: We don't attempt to test all rates as that would be time-consuming
            # and might cause issues with some audio servers
            logger.info(f"  Assuming support for common rates like 44100Hz and 48000Hz")
            
            # Add common rates that are typically supported with PipeWire/PulseAudio
            self.supported_rates = [8000, 11025, 16000, 22050, 32000, 44100, 48000]
            
        except Exception as e:
            logger.warning(f"Error detecting device capabilities: {str(e)}")
            # Fallback to safe defaults if detection fails
            self.default_rate = 44100
            self.supported_rates = [44100, 48000]
            
    def _adjust_sample_rate(self):
        """Adjust the sample rate if the preferred rate isn't supported"""
        # If preferred rate is in supported rates or we couldn't detect, keep it
        if not self.supported_rates or self.preferred_rate in self.supported_rates:
            return
            
        # If default rate is available, use that
        if self.default_rate:
            logger.warning(f"Preferred sample rate {self.preferred_rate}Hz is not supported. "
                          f"Using device default rate: {self.default_rate}Hz")
            self.rate = self.default_rate
            return
            
        # Otherwise find the closest supported rate
        closest_rate = min(self.supported_rates, key=lambda x: abs(x - self.preferred_rate))
        logger.warning(f"Preferred sample rate {self.preferred_rate}Hz is not supported. "
                      f"Using closest available rate: {closest_rate}Hz")
        self.rate = closest_rate

    async def _record_audio(self):
        """Record audio data from the selected PipeWire source"""
        source = await self._setup_pipewire()
        
        try:
            # Create a recording stream
            self.stream = self.pulse.stream_monitor_source_by_source_name(source.name)
            stream_id = self.stream.index
            logger.info(f"Started recording from source: {source.name} (stream ID: {stream_id})")
            
            last_data_time = time.time()
            consecutive_failures = 0
            frames = []
            frame_count = 0
            
            # Main recording loop
            while self.running:
                try:
                    # Read from the stream - this is blocking but with a timeout
                    data = self.pulse.read(stream_id, self.chunk_size)
                    
                    if data and len(data) > 0:
                        # Reset failure counter on success
                        consecutive_failures = 0
                        frames.append(data)
                        frame_count += 1
                        last_data_time = time.time()
                        
                        # If we've collected enough frames for a sample
                        if frame_count >= self.chunks_per_sample:
                            # Process the collected frames
                            self._process_frames(frames)
                            frames = []
                            frame_count = 0
                    else:
                        # No data received
                        consecutive_failures += 1
                        if consecutive_failures >= 5:
                            logger.warning(f"No audio data received for {time.time() - last_data_time:.1f} seconds")
                            consecutive_failures = 0
                            
                        # Check if we've been silent too long
                        if time.time() - last_data_time > 2.0:
                            logger.warning("Audio stream appears to be silent, checking connection...")
                            # Try to poke the connection to see if it's still alive
                            try:
                                self.pulse.source_info(source.name)
                                logger.info("Connection still alive, continuing to listen")
                                last_data_time = time.time()  # Reset the timer
                            except Exception as e:
                                logger.error(f"Source no longer available: {e}")
                                break  # Exit the loop to trigger a reconnection
                    
                    # Brief pause to prevent CPU overuse
                    await asyncio.sleep(0.01)
                    
                except Exception as e:
                    logger.error(f"Error during audio recording: {str(e)}")
                    consecutive_failures += 1
                    
                    # Break loop after too many errors to trigger reconnection
                    if consecutive_failures >= 3:
                        logger.warning("Too many consecutive errors, restarting stream")
                        break
                        
                    # Brief pause after error
                    await asyncio.sleep(0.5)
                    
        except Exception as e:
            logger.error(f"Error setting up recording: {str(e)}")
        finally:
            # Clean up the stream
            if self.stream:
                try:
                    self.pulse.stream_delete(self.stream.index)
                    self.stream = None
                except Exception as e:
                    logger.error(f"Error closing PipeWire stream: {str(e)}")

    def _process_frames(self, frames):
        """Process collected audio frames into a sample"""
        if not frames:
            return False
            
        try:
            # Convert frames to numpy array
            raw_audio = b''.join(frames)
            
            # Add to buffer
            logger.info(f"Adding audio chunk of {len(raw_audio)} bytes")
            self.audio_buffer.append(raw_audio)
            self.sample_ready.set()
            self.sample_ready.clear()
            
            # Update last sample time for watchdog
            self.last_sample_time = time.time()
            return True
        except Exception as e:
            logger.error(f"Error processing audio frames: {str(e)}")
            return False

    def _cleanup_pulse_client(self):
        """Clean up PipeWire/PulseAudio client and resources"""
        if self.pulse:
            try:
                if self.stream:
                    try:
                        self.pulse.stream_delete(self.stream.index)
                    except:
                        pass
                    self.stream = None
                    
                self.pulse.close()
            except Exception as e:
                logger.warning(f"Error cleaning up PipeWire client: {str(e)}")
            finally:
                self.pulse = None

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

        # Clean up PipeWire resources
        self._cleanup_pulse_client()
        
        # Clean up thread pool
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)

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

        # Find audio device using PipeWire/PulseAudio
        matches = self._find_audio_device()
        if len(matches) == 0:
            raise RuntimeError("No audio device available")
        self.device_index = matches[0]["index"]
        # Initialize components
        self._initialize_audio_components()

    def _find_audio_device(self, match_on: str = "alsa_input.usb-GeneralPlus_USB_Audio_Device-00.mono-fallback") -> list:
        matches = []
        try:
            # Use pulsectl (PipeWire's PulseAudio compatibility layer) to find devices
            with pulsectl.Pulse('device-finder') as pulse:
                sources = pulse.source_list()
                
                # Log all sources for debugging
                for i, source in enumerate(sources):
                    # Get default sample rate from the device
                    default_rate = source.sample_spec.rate if hasattr(source, 'sample_spec') else 44100
                    
                    device_info = {
                        "index": i,
                        "name": source.name,
                        "description": source.description,
                        "maxInputChannels": source.channel_count,
                        "defaultSampleRate": default_rate
                    }
                    logger.info(f"Found audio device: {device_info}")
                    
                    # Match based on name or description
                    if (match_on.lower() in source.name.lower() or 
                        match_on.lower() in source.description.lower()):
                        matches.append(device_info)
                        logger.info(f"Found a match!: {device_info}")
                
                # If no matches found, add default source
                if not matches and sources:
                    server_info = pulse.server_info()
                    default_source_name = server_info.default_source_name
                    for i, source in enumerate(sources):
                        if source.name == default_source_name:
                            # Get default sample rate
                            default_rate = source.sample_spec.rate if hasattr(source, 'sample_spec') else 44100
                            
                            device_info = {
                                "index": i,
                                "name": source.name,
                                "description": source.description,
                                "maxInputChannels": source.channel_count,
                                "defaultSampleRate": default_rate
                            }
                            matches.append(device_info)
                            logger.info(f"Using default source as fallback: {device_info}")
                            break
                        except Exception as e:
                            logger.error(f"Error finding audio devices with PipeWire: {e}")
            # As a last resort, if we can't find devices through PipeWire, try to return a default device
            matches.append({"index": 0, "name": "default", "maxInputChannels": 1, "defaultSampleRate": 44100})
            
        return matches

    def _initialize_audio_components(self):
        # Create and start recording loop with PipeWire
        logger.info(f"Initializing PipeWire recording loop with device index {self.device_index}")
        
        # Find matching device to get its capabilities
        matches = self._find_audio_device()
        device_rate = self.rate  # Default to our configured rate
        
        # If we have a match and it has a defined sample rate, use that instead
        if matches and "defaultSampleRate" in matches[0]:
            detected_rate = matches[0]["defaultSampleRate"]
            if detected_rate != self.rate:
                logger.info(f"Using detected sample rate from device: {detected_rate}Hz "
                           f"(instead of configured {self.rate}Hz)")
                device_rate = detected_rate
        
        self.recording_loop = PipeWireRecordingLoop(
            device_index=self.device_index,
            rate=device_rate,
            channels=self.channels,
            chunk_size=1024,
        )
        self.recording_loop.start(sample_duration=self.sample_duration)
    
        try:
            # Wait for first audio sample
            logger.info("Waiting for initial audio sample...")
            if not self.recording_loop.wait_for_data(timeout=15):  # Increased timeout for PipeWire initialization
                self.recording_loop.stop()
                raise RuntimeError("Could not record initial audio sample from PipeWire")
            
            logger.info("Initial audio sample received, starting processing loop")
            
            # Create and start processing loop
            self.processing_loop = ProcessingLoop(
                rate=self.rate, mfcc_count=self.mfcc_count, n_fft=2048
            )
            self.processing_loop.start(self.recording_loop.get_audio_data)
        
            # Initialize buffer for processed data
            self.mfcc_buffer = deque(maxlen=self.buffer_size)
            self.spectrum_buffer = deque(maxlen=self.buffer_size)
        
            # Wait for initial processing
            logger.info("Waiting for initial audio processing...")
            if not self.processing_loop.wait_for_processing(timeout=20):
                self.recording_loop.stop()
                self.processing_loop.stop()
                raise RuntimeError("Could not process initial audio sample")
                
            logger.info("Audio components successfully initialized with PipeWire")
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
            
            # Reinitialize recording loop with PipeWire
            self.recording_loop = PipeWireRecordingLoop(
                device_index=self.device_index,
                rate=self.rate,
                channels=self.channels,
                chunk_size=1024,
            )
            self.recording_loop.start(sample_duration=self.sample_duration)
            
            # Wait for first audio sample
            if not self.recording_loop.wait_for_data(timeout=10):  # Increased timeout for PipeWire
                logger.error("Could not record audio during recovery")
                return
                
            # Recreate processing loop
            self.processing_loop = ProcessingLoop(
                rate=self.rate, mfcc_count=self.mfcc_count, n_fft=2048
            )
            self.processing_loop.start(self.recording_loop.get_audio_data)
            
            # Wait for initial processing
            if not self.processing_loop.wait_for_processing(timeout=10):  # Increased timeout for processing
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
