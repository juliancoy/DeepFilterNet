"""SoundFile/SciPy audio I/O for runtimes without a matching torchaudio wheel."""
from dataclasses import dataclass
from math import gcd
from types import SimpleNamespace
import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly


@dataclass
class AudioMetaData:
    sample_rate: int
    num_frames: int
    num_channels: int
    bits_per_sample: int
    encoding: str


def info(path, **kwargs):
    value = sf.info(path)
    bits = int(value.subtype.split('_')[-1]) if value.subtype.split('_')[-1].isdigit() else 0
    return AudioMetaData(value.samplerate, value.frames, value.channels, bits, value.subtype)


def load(path, frame_offset=0, num_frames=-1, channels_first=True, **kwargs):
    audio, rate = sf.read(path, start=frame_offset, frames=num_frames,
                          dtype='float32', always_2d=True)
    return torch.from_numpy(audio.T.copy() if channels_first else audio), rate


def save(path, audio, sample_rate, channels_first=True, bits_per_sample=16, **kwargs):
    data = audio.detach().cpu().numpy()
    if channels_first:
        data = data.T
    subtype = {16: 'PCM_16', 24: 'PCM_24', 32: 'PCM_32'}.get(bits_per_sample, 'PCM_16')
    sf.write(path, data, sample_rate, subtype=subtype)


def resample(audio, orig_freq, new_freq, **kwargs):
    if orig_freq == new_freq:
        return audio
    factor = gcd(int(orig_freq), int(new_freq))
    data = resample_poly(audio.detach().cpu().numpy(), int(new_freq)//factor,
                         int(orig_freq)//factor, axis=-1)
    return torch.from_numpy(np.ascontiguousarray(data)).to(device=audio.device, dtype=audio.dtype)


functional = SimpleNamespace(resample=resample)
