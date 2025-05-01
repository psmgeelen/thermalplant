from collections import deque
import asyncio
import numpy as np
import librosa
import pyaudio
import pulsectl
import logging
import time
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from .utils import get_logger


class AudioHandlerSettings(BaseModel):
    """Configuration settings for AudioHandler"""
    sample_duration: float = Field(..., gt=0, description="Duration of audio sample in seconds")
    mfcc_count: int = Field(..., gt=0, description="Number of MFCC coefficients to extract")
    buffer_size: int = Field(..., gt=0, description="Size of audio buffer")


class IntegratedAudioProcessor:
    """Combined audio recording and processing system"""
    
    def __init__(self, rate: int = 44100, channels: int = 1,
                 device_name: str = "default", mfcc_count: int = 13,
                 buffer_size: int = 3, n_fft: int = 2048):
        self.logger = get_logger(__name__)
        self.rate = rate
        self.channels = channels
        self.device_name = device_name
        self.mfcc_count = mfcc_count
        self.n_fft = n_fft
        
        # Buffers
        self.raw_buffer = deque(maxlen=buffer_size)
        self.features_buffer = deque(maxlen=buffer_size)
        
        # Audio handling
        self.chunk_size = int(rate)  # 1-second chunks
        self.pa = None
        self.pulse = None
        self.stream = None
        
        # State
        self.running = False
        self._processing_task = None

    async def start(self):
        """Initialize and start audio processing"""
        try:
            # Initialize audio
            self.pulse = pulsectl.Pulse('device-detector')
            self.pa = pyaudio.PyAudio()
            
            # Find and set up device
            await self._setup_audio_device()
            
            self.running = True
            self._processing_task = asyncio.create_task(self._process_loop())
            self.logger.info("Audio processor started")
            
        except Exception as e:
            self.logger.error(f"Failed to start audio processor: {e}")
            await self.stop()
            raise

    async def _setup_audio_device(self):
        """Set up audio device and stream"""
        sources = self.pulse.source_list()
        self.logger.info("Available audio sources:")
        for source in sources:
            self.logger.info(f"  - {source.name}")

        # Try to find specified device
        device_index = None
        for i in range(self.pa.get_device_count()):
            device_info = self.pa.get_device_info_by_index(i)
            if (self.device_name in device_info['name'] and 
                device_info['maxInputChannels'] > 0):
                device_index = i
                break

        if device_index is None:
            self.logger.warning(f"Device '{self.device_name}' not found, using default")
            device_index = self.pa.get_default_input_device_info()['index']

        self.stream = self.pa.open(
            format=pyaudio.paFloat32,
            channels=self.channels,
            rate=self.rate,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=self.chunk_size,
            stream_callback=self._audio_callback
        )
        
        self.logger.info(f"Audio stream opened: device_index={device_index}, "
                        f"rate={self.rate}, channels={self.channels}")

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Handle incoming audio data"""
        if status:
            self.logger.warning(f"Audio callback status: {status}")
        
        if self.running:
            try:
                audio_data = np.frombuffer(in_data, dtype=np.float32)
                self.raw_buffer.append(audio_data)
                
                # Log audio levels
                rms = np.sqrt(np.mean(np.square(audio_data)))
                self.logger.debug(f"Audio RMS level: {rms:.6f}")
                
            except Exception as e:
                self.logger.error(f"Error in audio callback: {e}")
                
        return (None, pyaudio.paContinue)

    async def _process_loop(self):
        """Process audio data from the buffer"""
        while self.running:
            try:
                if len(self.raw_buffer) > 0:
                    audio_data = self.raw_buffer[-1]
                    
                    features = {
                        'mfcc': await self._compute_mfcc(audio_data),
                        'spectrum': await self._compute_spectrum(audio_data),
                        'timestamp': time.time(),
                        'status': 'ok'
                    }
                    
                    self.features_buffer.append(features)
                    
                await asyncio.sleep(0.1)
                    
            except Exception as e:
                self.logger.error(f"Processing error: {e}")
                await asyncio.sleep(1)

    async def _compute_mfcc(self, audio_np: np.ndarray) -> Dict[str, float]:
        """Compute MFCC features"""
        try:
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

    async def _compute_spectrum(self, audio_np: np.ndarray) -> Dict[str, float]:
        """Compute frequency spectrum"""
        try:
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

    def get_latest_features(self) -> Dict[str, Any]:
        """Get the most recent processed features"""
        if len(self.features_buffer) > 0:
            return self.features_buffer[-1]
        return {
            'mfcc': {},
            'spectrum': {},
            'timestamp': time.time(),
            'status': 'no_data'
        }

    async def stop(self):
        """Stop processing and cleanup"""
        self.running = False
        
        if self._processing_task:
            try:
                await self._processing_task
            except Exception as e:
                self.logger.error(f"Error stopping processing task: {e}")

        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception as e:
                self.logger.error(f"Error closing audio stream: {e}")

        if self.pa:
            try:
                self.pa.terminate()
            except Exception as e:
                self.logger.error(f"Error terminating PyAudio: {e}")

        if self.pulse:
            try:
                self.pulse.close()
            except Exception as e:
                self.logger.error(f"Error closing Pulse connection: {e}")

        self.logger.info("Audio processor stopped")


class AudioHandler:
    """API-compatible audio handler using integrated processor"""
    
    def __init__(self, rate: int, channels: int, sample_duration: float,
                 mfcc_count: int, buffer_size: int = 3):
        self.logger = get_logger(__name__)
        self.settings = AudioHandlerSettings(
            sample_duration=sample_duration,
            mfcc_count=mfcc_count,
            buffer_size=buffer_size
        )
        
        self.processor = IntegratedAudioProcessor(
            rate=rate,
            channels=channels,
            mfcc_count=mfcc_count,
            buffer_size=buffer_size
        )
        
        # Start the processor
        asyncio.create_task(self.processor.start())

    def read_mfcc(self) -> Dict[str, float]:
        """Get latest MFCC features"""
        try:
            return self.processor.get_latest_features().get('mfcc', {})
        except Exception as e:
            self.logger.error(f"Error reading MFCC: {e}")
            return {'error': str(e)}

    def read_spectrum(self) -> Dict[str, float]:
        """Get latest spectrum data"""
        try:
            return self.processor.get_latest_features().get('spectrum', {})
        except Exception as e:
            self.logger.error(f"Error reading spectrum: {e}")
            return {'error': str(e)}

    def read_all_audio(self) -> Dict[str, Any]:
        """Get all audio features"""
        try:
            features = self.processor.get_latest_features()
            return {
                'mfcc': features.get('mfcc', {}),
                'spectrum': features.get('spectrum', {}),
                'timestamp': features.get('timestamp', time.time()),
                'status': features.get('status', 'error')
            }
        except Exception as e:
            self.logger.error(f"Error reading audio data: {e}")
            return {
                'mfcc': {},
                'spectrum': {},
                'timestamp': time.time(),
                'status': 'error',
                'error': str(e)
            }

    async def close(self):
        """Cleanup resources"""
        try:
            await self.processor.stop()
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")