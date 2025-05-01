import asyncio
import numpy as np
import librosa
import logging
import time
from typing import Optional, Dict, Any, List
import pulsectl_async
from pydantic import BaseModel, Field
from utils import get_logger


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
    """Handles audio recording using PipeWire/PulseAudio"""

    def __init__(self, device_index: int, rate: int, channels: int, chunk_size: int = 1024):
        super().__init__("PipewireRecordingLoop", get_logger(__name__))
        self.device_index = device_index
        self.rate = rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.chunks_per_sample = int(rate / chunk_size)  # For 1-second samples
        self.audio_queue = asyncio.Queue(maxsize=10)
        self.pulse = None
        self.stream = None
        self.last_sample_time = time.time()
        self.max_silence_duration = 2.0  # Maximum time to wait for audio before reconnecting

    async def _run(self):
        """Main recording loop"""
        try:
            async with pulsectl_async.PulseAsync('audio-recorder') as self.pulse:
                while self.running:
                    try:
                        await self._setup_recording()
                        await self._record_loop()
                    except Exception as e:
                        self.logger.error(f"Recording error, will retry: {e}")
                        await asyncio.sleep(1)
                    finally:
                        await self._cleanup()
        except Exception as e:
            self.logger.error(f"Fatal recording error: {e}")

    async def _setup_recording(self):
        """Setup PipeWire recording stream"""
        if self.pulse is None:
            raise RuntimeError("PulseAudio client not initialized")

        # Get device info
        server_info = await self.pulse.server_info()
        sources = await self.pulse.source_list()

        # Find the requested device
        source = next((s for s in sources if s.index == self.device_index), None)
        if not source:
            raise RuntimeError(f"Audio device with index {self.device_index} not found")

        # Create recording stream
        self.stream = await self.pulse.stream_create(
            name="audio-recorder",
            rate=self.rate,
            channels=self.channels,
            format="s16le",
            source=source.name
        )

        # Set buffer attributes
        await self.pulse.stream_set_buffer_attr(
            self.stream,
            fragsize=self.chunk_size * 2
        )

    async def _record_loop(self):
        """Main recording loop"""
        frames: List[bytes] = []
        frame_count = 0
        consecutive_failures = 0

        while self.running:
            try:
                data = await self._record_chunk()
                if data:
                    consecutive_failures = 0
                    frames.append(data)
                    frame_count += 1
                    self.last_sample_time = time.time()

                    if frame_count >= self.chunks_per_sample:
                        # Combine frames and put in queue
                        complete_sample = b''.join(frames)
                        await self.audio_queue.put(complete_sample)
                        frames = []
                        frame_count = 0
                else:
                    consecutive_failures += 1
                    if consecutive_failures >= 3:
                        raise RuntimeError("Multiple consecutive recording failures")

                await asyncio.sleep(0.001)  # Tiny sleep to prevent CPU overuse

            except Exception as e:
                self.logger.error(f"Recording error: {e}")
                break

    async def _record_chunk(self):
        """Record a single chunk of audio data"""
        try:
            if not self.stream:
                raise RuntimeError("Recording stream not initialized")

            data = await self.pulse.stream_read(self.stream, self.chunk_size)
            return data

        except Exception as e:
            self.logger.error(f"Error recording audio chunk: {e}")
            return None

    async def _cleanup(self):
        """Cleanup PipeWire resources"""
        try:
            if self.stream:
                await self.pulse.stream_delete(self.stream)
                self.stream = None
        except Exception as e:
            self.logger.error(f"Error cleaning up PipeWire resources: {e}")


class AudioProcessingLoop(AsyncComponent):
    """Handles audio processing into MFCC and spectrum features"""

    def __init__(self, rate: int, mfcc_count: int, n_fft: int = 2048):
        super().__init__("AudioProcessingLoop", get_logger(__name__))
        self.rate = rate
        self.mfcc_count = mfcc_count
        self.n_fft = n_fft
        self.results_queue = asyncio.Queue(maxsize=3)
        self.audio_queue = None  # Set by AudioHandler during initialization

    async def _run(self):
        """Main processing loop"""
        if not self.audio_queue:
            raise RuntimeError("Audio queue not set")

        while self.running:
            try:
                # Get audio data from queue
                audio_data = await self.audio_queue.get()

                # Convert to numpy array
                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)

                # Process features
                mfcc = await self._compute_mfcc(audio_np)
                spectrum = await self._compute_spectrum(audio_np)

                # Package results
                result = {
                    'mfcc': mfcc,
                    'spectrum': spectrum,
                    'timestamp': time.time()
                }

                # Put results in queue, dropping oldest if full
                try:
                    await self.results_queue.put_nowait(result)
                except asyncio.QueueFull:
                    _ = await self.results_queue.get()  # Remove oldest
                    await self.results_queue.put(result)

            except Exception as e:
                self.logger.error(f"Processing error: {e}")
                await asyncio.sleep(0.1)

    async def _compute_mfcc(self, audio_np: np.ndarray) -> dict:
        """Compute MFCC features from audio data"""
        try:
            # Normalize audio
            audio_np = audio_np / np.iinfo(np.int16).max

            # Compute MFCCs
            mfccs = librosa.feature.mfcc(
                y=audio_np,
                sr=self.rate,
                n_mfcc=self.mfcc_count,
                n_fft=self.n_fft
            )

            # Create frequency labels (mel scale)
            mel_freqs = librosa.mel_frequencies(
                n_mels=self.mfcc_count,
                fmin=0,
                fmax=self.rate / 2
            )

            # Package results with labels
            return {
                f"mfcc_{i}_{freq:.0f}hz": float(mfcc)
                for i, (freq, mfcc) in enumerate(zip(mel_freqs, mfccs.mean(axis=1)))
            }

        except Exception as e:
            self.logger.error(f"Error computing MFCCs: {e}")
            return {}

    async def _compute_spectrum(self, audio_np: np.ndarray) -> dict:
        """Compute frequency spectrum from audio data"""
        try:
            # Normalize audio
            audio_np = audio_np / np.iinfo(np.int16).max

            # Compute spectrogram
            D = librosa.stft(audio_np, n_fft=self.n_fft)
            spectrum = np.abs(D)

            # Convert to dB scale
            spectrum_db = librosa.amplitude_to_db(spectrum, ref=np.max)

            # Get frequency bins
            freqs = librosa.fft_frequencies(sr=self.rate, n_fft=self.n_fft)

            # Average across time
            mean_spectrum = spectrum_db.mean(axis=1)

            # Package results with frequency labels
            return {
                f"band_{freq:.0f}hz": float(power)
                for freq, power in zip(freqs, mean_spectrum)
            }

        except Exception as e:
            self.logger.error(f"Error computing spectrum: {e}")
            return {}


class AudioWatchdog(AsyncComponent):
    """Monitors audio components and handles recovery"""

    def __init__(self, check_interval: float = 1.0):
        super().__init__("AudioWatchdog", get_logger(__name__))
        self.check_interval = check_interval
        self.components = {}

    def register_component(self, name: str, component: AsyncComponent):
        """Register a component for monitoring"""
        self.components[name] = component

    async def _run(self):
        """Main watchdog loop"""
        while self.running:
            for name, component in self.components.items():
                try:
                    # Check component health and restart if needed
                    if not component.running:
                        self.logger.warning(f"{name} not running, attempting restart")
                        await component.start()
                except Exception as e:
                    self.logger.error(f"Error monitoring {name}: {e}")
            await asyncio.sleep(self.check_interval)


class AudioHandler:
    """Main orchestrator for audio recording and processing"""

    def __init__(self, device_index: int, rate: int, channels: int,
                 mfcc_count: int, buffer_size: int = 3):
        self.logger = get_logger(__name__)

        # Initialize components
        self.recording_loop = PipewireRecordingLoop(
            device_index=device_index,
            rate=rate,
            channels=channels
        )

        self.processing_loop = AudioProcessingLoop(
            rate=rate,
            mfcc_count=mfcc_count
        )

        # Connect components
        self.processing_loop.audio_queue = self.recording_loop.audio_queue

        # Setup watchdog
        self.watchdog = AudioWatchdog()
        self.watchdog.register_component("recording", self.recording_loop)
        self.watchdog.register_component("processing", self.processing_loop)

    async def start(self):
        """Start all audio components"""
        try:
            await self.watchdog.start()
            await self.recording_loop.start()
            await self.processing_loop.start()
            self.logger.info("Audio handler started successfully")
        except Exception as e:
            self.logger.error(f"Failed to start audio handler: {e}")
            await self.stop()
            raise

    async def stop(self):
        """Stop all audio components gracefully"""
        try:
            await self.processing_loop.stop()
            await self.recording_loop.stop()
            await self.watchdog.stop()
            self.logger.info("Audio handler stopped successfully")
        except Exception as e:
            self.logger.error(f"Error stopping audio handler: {e}")
            raise

    async def get_latest_features(self) -> Dict[str, Any]:
        """Get the latest processed audio features"""
        try:
            if not self.processing_loop.results_queue.empty():
                result = await self.processing_loop.results_queue.get()
                return {
                    'mfcc': result.get('mfcc', {}),
                    'spectrum': result.get('spectrum', {}),
                    'timestamp': result.get('timestamp', 0)
                }
            else:
                return {
                    'mfcc': {},
                    'spectrum': {},
                    'timestamp': 0,
                    'status': 'no_data'
                }
        except Exception as e:
            self.logger.error(f"Error getting audio features: {e}")
            raise


# Configuration validation model
class AudioHandlerConfig(BaseModel):
    """Configuration settings for AudioHandler"""
    sample_duration: float = Field(..., gt=0)
    mfcc_count: int = Field(..., gt=0)
    buffer_size: int = Field(..., gt=0)