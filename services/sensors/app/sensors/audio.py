import asyncio
import numpy as np
import librosa
import logging
import time
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, validator
from .utils import get_logger
import numpy as np
import time
import pyaudio
import pulsectl
import numpy as np
from typing import Optional, Dict


class AsyncComponent:
    """Base class for async components with lifecycle management and logging"""

    def __init__(self, name: str, logger: logging.Logger):
        self.name = name
        self.logger = logger
        self.running = False
        self._task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

    async def start(self):
        """Start the async component"""
        if self.running:
            return
        self.running = True
        self._stop_event.clear()
        self._task = asyncio.create_task(self._run())
        self.logger.info(f"{self.name} started")

    async def stop(self):
        """Stop the async component gracefully"""
        if not self.running:
            return
        self.running = False
        self._stop_event.set()
        if self._task:
            await self._task
            self._task = None
        self.logger.info(f"{self.name} stopped")

    async def _run(self):
        """Main execution loop to be implemented by subclasses"""
        raise NotImplementedError()



class PipewireRecordingLoop(AsyncComponent):
    """Handles audio recording using PyAudio with PulseAudio/Pipewire device detection"""

    def __init__(self, device_name: str = "USB Audio Device", rate: int = 44100, channels: int = 1):
        super().__init__("PipewireRecordingLoop", get_logger(__name__))
        self.device_name = device_name
        self.rate = rate
        self.channels = channels  # We want mono
        self.audio_queue = asyncio.Queue(maxsize=10)
        self.pulse = None
        self.pa = None
        self.stream = None
        self.device_index = None
        self.chunk_size = int(self.rate)  # 1 second chunks

    async def _setup_recording(self):
        """Setup recording by finding the correct device and initializing PyAudio stream"""
        if not self.pulse:
            raise RuntimeError("PulseAudio client not initialized")

        # Get PulseAudio sources
        sources = self.pulse.source_list()
        self.logger.info(f"Found the following PulseAudio sources: {sources}")

        # First, look for the mono input device
        pulse_source = None
        for source in sources:
            # Look specifically for USB Audio Device that's mono
            if (self.device_name in source.name and 
                'mono' in source.name.lower() and 
                source.channels == 1):
                pulse_source = source
                self.logger.info(f"Found matching mono source: {source}")
                break
        
        # If no mono device found, try to find any matching device
        if not pulse_source:
            for source in sources:
                if self.device_name in source.name:
                    pulse_source = source
                    self.logger.info(f"Found matching source (non-mono): {source}")
                    self.logger.warning("Using non-mono device, audio will be downmixed to mono")
                    break

        if not pulse_source:
            raise RuntimeError(f"Audio device containing '{self.device_name}' not found")

        # Store the actual number of channels from the source
        source_channels = pulse_source.channels
        self.logger.info(f"Source has {source_channels} channels, will {'downmix to' if source_channels > 1 else 'use as'} mono")

        # Find PyAudio device index
        device_count = self.pa.get_device_count()
        for i in range(device_count):
            device_info = self.pa.get_device_info_by_index(i)
            self.logger.info(f"Checking device {i}: {device_info}")
            # Match by name and input capability
            if (pulse_source.name in device_info.get('name', '') and 
                device_info.get('maxInputChannels', 0) > 0):
                self.device_index = i
                break

        if self.device_index is None:
            raise RuntimeError(f"Could not find PyAudio device for {pulse_source.name}")

        # Create PyAudio stream
        self.stream = self.pa.open(
            format=pyaudio.paInt16,
            channels=source_channels,  # Use source channels
            rate=self.rate,
            input=True,
            input_device_index=self.device_index,
            frames_per_buffer=self.chunk_size
        )

        self.logger.info("Stream opened successfully")

        # Try to move the PyAudio stream to the correct PulseAudio source
        try:
            source_outputs = self.pulse.source_output_list()
            for output in source_outputs:
                if 'PyAudio' in output.name:
                    self.logger.info(f"Moving PyAudio output to source: {pulse_source.name}")
                    self.pulse.source_output_move(output.index, pulse_source.index)
                    break
        except Exception as e:
            self.logger.warning(f"Could not set PulseAudio source: {e}")

        self.logger.info("Recording setup completed successfully")

class AudioProcessingLoop(AsyncComponent):
    """Handles audio processing into MFCC and spectrum features"""

    def __init__(self, rate: int, mfcc_count: int, n_fft: int = 2048):
        super().__init__("AudioProcessingLoop", get_logger(__name__))
        self.rate = rate
        self.mfcc_count = mfcc_count
        self.n_fft = n_fft
        self.results_queue = asyncio.Queue(maxsize=3)
        self.audio_queue = None

    async def _run(self):
        """Main processing loop"""
        if not self.audio_queue:
            raise RuntimeError("Audio queue not set")

        while self.running:
            try:
                # Use wait_for to prevent indefinite blocking
                audio_data = await asyncio.wait_for(self.audio_queue.get(), timeout=1.0)
                if audio_data is None:
                    continue

                # Process audio data in a separate thread to not block the event loop
                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
                
                # Run CPU-intensive operations in a thread pool
                mfcc = await asyncio.to_thread(self._compute_mfcc, audio_np)
                spectrum = await asyncio.to_thread(self._compute_spectrum, audio_np)

                result = {
                    'mfcc': mfcc,
                    'spectrum': spectrum,
                    'timestamp': time.time()
                }

                try:
                    await asyncio.wait_for(
                        self.results_queue.put(result),
                        timeout=0.1
                    )
                except asyncio.TimeoutError:
                    # If queue is full, remove oldest item and try again
                    if not self.results_queue.empty():
                        _ = await self.results_queue.get()
                    await self.results_queue.put(result)

            except asyncio.TimeoutError:
                # No data available, continue waiting
                continue
            except Exception as e:
                self.logger.error(f"Processing error: {e}")
                await asyncio.sleep(0.1)

    def _compute_mfcc(self, audio_np: np.ndarray) -> Dict[str, float]:
        """Compute MFCC features"""
        try:
            audio_np = audio_np / np.iinfo(np.int16).max
            mfccs = librosa.feature.mfcc(
                y=audio_np,
                sr=self.rate,
                n_mfcc=self.mfcc_count,
                n_fft=self.n_fft
            )

            mel_freqs = librosa.mel_frequencies(
                n_mels=self.mfcc_count,
                fmin=0,
                fmax=self.rate / 2
            )

            return {
                f"mfcc_{i}_{freq:.0f}hz": float(mfcc)
                for i, (freq, mfcc) in enumerate(zip(mel_freqs, mfccs.mean(axis=1)))
            }
        except Exception as e:
            self.logger.error(f"MFCC computation error: {e}")
            return {}

    def _compute_spectrum(self, audio_np: np.ndarray) -> Dict[str, float]:
        """Compute frequency spectrum"""
        try:
            audio_np = audio_np / np.iinfo(np.int16).max
            D = librosa.stft(audio_np, n_fft=self.n_fft)
            spectrum = np.abs(D)
            spectrum_db = librosa.amplitude_to_db(spectrum, ref=np.max)
            freqs = librosa.fft_frequencies(sr=self.rate, n_fft=self.n_fft)
            mean_spectrum = spectrum_db.mean(axis=1)

            return {
                f"band_{freq:.0f}hz": float(power)
                for freq, power in zip(freqs, mean_spectrum)
            }
        except Exception as e:
            self.logger.error(f"Spectrum computation error: {e}")
            return {}

class AudioHandler:
    """Main audio processing handler"""

    def __init__(self, rate: int, channels: int, sample_duration: float,
                 mfcc_count: int, buffer_size: int = 3):
        self.logger = get_logger(__name__)
        self.settings = AudioHandlerSettings(
            sample_duration=sample_duration,
            mfcc_count=mfcc_count,
            buffer_size=buffer_size
        )

        # Initialize components
        self.recording_loop = PipewireRecordingLoop(
            rate=rate,
            channels=channels
        )

        self.processing_loop = AudioProcessingLoop(
            rate=rate,
            mfcc_count=mfcc_count
        )

        # Connect components
        self.processing_loop.audio_queue = self.recording_loop.audio_queue

        # Start components
        asyncio.create_task(self._start())

    async def _start(self):
        """Start audio processing components"""
        try:
            await self.recording_loop.start()
            await self.processing_loop.start()
            self.logger.info("Audio handler started successfully")
        except Exception as e:
            self.logger.error(f"Failed to start audio handler: {e}")
            await self._stop()
            raise

    async def _stop(self):
        """Stop audio processing components"""
        try:
            await self.processing_loop.stop()
            await self.recording_loop.stop()
            self.logger.info("Audio handler stopped successfully")
        except Exception as e:
            self.logger.error(f"Error stopping audio handler: {e}")
            raise

    def close(self):
        """Clean up resources (used during shutdown)"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                future = asyncio.run_coroutine_threadsafe(self._stop(), loop)
                future.result(timeout=5)
            else:
                loop.run_until_complete(self._stop())
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

    def read_mfcc(self) -> Dict[str, float]:
        """Get latest MFCC features"""
        try:
            loop = asyncio.get_event_loop()
            if not self.processing_loop.results_queue.empty():
                result = loop.run_until_complete(
                    self.processing_loop.results_queue.get())
                return result.get('mfcc', {})
            return {}
        except Exception as e:
            self.logger.error(f"Error reading MFCC: {e}")
            return {'error': str(e)}

    def read_spectrum(self) -> Dict[str, float]:
        """Get latest spectrum data"""
        try:
            loop = asyncio.get_event_loop()
            if not self.processing_loop.results_queue.empty():
                result = loop.run_until_complete(
                    self.processing_loop.results_queue.get())
                return result.get('spectrum', {})
            return {}
        except Exception as e:
            self.logger.error(f"Error reading spectrum: {e}")
            return {'error': str(e)}

    def read_all_audio(self) -> Dict[str, Any]:
        """Get all audio features"""
        try:
            loop = asyncio.get_event_loop()
            if not self.processing_loop.results_queue.empty():
                result = loop.run_until_complete(
                    self.processing_loop.results_queue.get())
                return {
                    'mfcc': result.get('mfcc', {}),
                    'spectrum': result.get('spectrum', {}),
                    'timestamp': result.get('timestamp', time.time()),
                    'status': 'ok'
                }
            return {
                'mfcc': {},
                'spectrum': {},
                'timestamp': time.time(),
                'status': 'no_data'
            }
        except Exception as e:
            self.logger.error(f"Error reading audio data: {e}")
            return {
                'error': str(e),
                'status': 'error'
            }
            
class AudioHandlerSettings(BaseModel):
    """Configuration settings for AudioHandler"""
    sample_duration: float = Field(..., gt=0, description="Duration of audio sample in seconds")
    mfcc_count: int = Field(..., gt=0, description="Number of MFCC coefficients to extract")
    buffer_size: int = Field(..., gt=0, description="Size of audio buffer")