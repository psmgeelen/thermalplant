from collections import deque
import asyncio
import numpy as np
import torch
import torchaudio
import pyaudio
import pulsectl
import logging
import time
from typing import Dict, Any
from pydantic import BaseModel, Field
from .utils import get_logger


class AudioHandlerSettings(BaseModel):
    sample_duration: float = Field(1.0, gt=0)
    mfcc_count: int = Field(50, gt=0)
    buffer_size: int = Field(3, gt=0)
    n_bands: int = Field(50, gt=0)


class IntegratedAudioProcessor:
    def __init__(self, rate: float = 44100, channels: int = 1,
                 device_name: str = "USB", mfcc_count: int = 50,
                 buffer_size: int = 3, n_fft: int = 2048, n_bands: int = 50):
        self.logger = get_logger(__name__)
        self.rate = rate
        self.channels = channels
        self.device_name = device_name
        self.mfcc_count = mfcc_count
        self.n_fft = n_fft
        self.n_bands = n_bands

        self.raw_buffer = deque(maxlen=buffer_size)
        self.features_buffer = deque(maxlen=buffer_size)

        self.chunk_size = int(rate)
        self.pa = None
        self.pulse = None
        self.stream = None

        self.running = False
        self._processing_task = None

        # Torch transforms
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=self.rate,
            n_mfcc=self.mfcc_count,
            melkwargs={"n_fft": self.n_fft, "hop_length": 512}
        )
        self.spectrogram_transform = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft,
            power=2.0
        )

    async def start(self):
        try:
            self.pulse = pulsectl.Pulse('device-detector')
            self.pa = pyaudio.PyAudio()
            await self._setup_audio_device()

            self.running = True
            self._processing_task = asyncio.create_task(self._process_loop())
            self.logger.info("Audio processor started")

        except Exception as e:
            self.logger.error(f"Failed to start audio processor: {e}")
            await self.stop()
            raise

    async def _setup_audio_device(self):
        sources = self.pulse.source_list()
        self.logger.info("Available audio sources:")
        for source in sources:
            self.logger.info(f"  - {source.name}")

        device_index = None
        for i in range(self.pa.get_device_count()):
            device_info = self.pa.get_device_info_by_index(i)
            if (self.device_name in device_info['name'] and
                device_info['maxInputChannels'] > 0):
                self.logger.info(f"found match with: {device_info}")
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
        if status:
            self.logger.warning(f"Audio callback status: {status}")

        if self.running:
            try:
                audio_data = np.frombuffer(in_data, dtype=np.float32)
                self.raw_buffer.append(audio_data)
                rms = np.sqrt(np.mean(np.square(audio_data)))
                self.logger.debug(f"Audio RMS level: {rms:.6f}")
            except Exception as e:
                self.logger.error(f"Error in audio callback: {e}")

        return (None, pyaudio.paContinue)

    async def _process_loop(self):
        while self.running:
            try:
                if self.raw_buffer:
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
        try:
            waveform = torch.tensor(audio_np, dtype=torch.float32).unsqueeze(0)
            mfcc = self.mfcc_transform(waveform).mean(dim=-1).squeeze(0)

            mel_freqs = torch.linspace(0, self.rate / 2, steps=self.mfcc_count)

            return {
                f"mfcc_{i}_{freq.item():.0f}hz": float(mfcc[i].item())
                for i, freq in enumerate(mel_freqs)
            }

        except Exception as e:
            self.logger.error(f"MFCC computation error: {e}")
            return {}

    async def _compute_spectrum(self, audio_np: np.ndarray) -> Dict[str, float]:
        try:
            waveform = torch.tensor(audio_np, dtype=torch.float32).unsqueeze(0)
            spectrogram = self.spectrogram_transform(waveform)[0]
            spectrogram_db = 10 * torch.log10(spectrogram + 1e-10)

            freqs = torch.fft.rfftfreq(self.n_fft, 1 / self.rate)
            min_freq = 20
            max_freq = self.rate / 2
            band_edges = torch.logspace(
                start=torch.log10(torch.tensor(min_freq, dtype=torch.float32)),
                end=torch.log10(torch.tensor(max_freq, dtype=torch.float32)),
                steps=self.n_bands + 1
            )

            result = {}
            for i in range(self.n_bands):
                lower_freq = band_edges[i].item()
                upper_freq = band_edges[i + 1].item()

                mask = (freqs >= lower_freq) & (freqs < upper_freq)
                indices = mask.nonzero(as_tuple=True)[0]

                if len(indices) > 0:
                    band_power = spectrogram_db[indices, :].mean().item()
                    label = f"spectrum_{i}_{lower_freq:.0f}hz_{upper_freq:.0f}hz"
                    result[label] = band_power

            return result

        except Exception as e:
            self.logger.error(f"Spectrum computation error: {e}")
            return {}

    def get_latest_features(self) -> Dict[str, Any]:
        if self.features_buffer:
            return self.features_buffer[-1]
        return {
            'mfcc': {},
            'spectrum': {},
            'timestamp': time.time(),
            'status': 'no_data'
        }

    async def stop(self):
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
                 mfcc_count: int, buffer_size: int = 3, n_bands: int = 50):
        self.logger = get_logger(__name__)
        self.settings = AudioHandlerSettings(
            sample_duration=sample_duration,
            mfcc_count=mfcc_count,
            buffer_size=buffer_size,
            n_bands=n_bands
        )
        
        self.processor = IntegratedAudioProcessor(
            rate=rate,
            channels=channels,
            mfcc_count=mfcc_count,
            buffer_size=buffer_size,
            n_bands=n_bands
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
