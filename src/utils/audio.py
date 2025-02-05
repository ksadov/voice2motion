from pathlib import Path
import torch
import torchaudio
import numpy as np
import torch.nn.functional as F

from functools import lru_cache
from numpy.typing import NDArray
from typing import Optional, Union

from src.utils.constants import (
    SAMPLE_RATE,
    N_FFT,
    HOP_LENGTH,
    N_SAMPLES,
    MEL_FILTER_PATH,
)


def load_audio_from_video(video_path: Path) -> NDArray[np.float32]:
    """
    Load the audio from a video file.

    Args:
        video_path (Path): Path to the video file.

    Returns:
        NDArray[np.float32]: A single-channel SAPLE_RATE Hz audio signal of
            shape (n_samples,)
    """
    audio, sr = torchaudio.load(video_path, normalize=True)
    # make single channel
    audio = audio.mean(dim=0, keepdim=True)
    # resample to 16 kHz
    audio = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)(audio)
    return audio[0].numpy()


def trim_audio(
    array: np.array,
    start_time: float,
    end_time: float,
    sample_rate: int = SAMPLE_RATE,
):
    """
    Trim the audio file base on the start and end time.

    Args:
        array (np.array): The audio array.
        start_time (float): The start time in seconds.
        end_time (float): The end time in seconds.
        sample_rate (int): The sample rate of the audio.

    Returns:
        np.array: The trimmed audio array.
    """
    start_frame = int(sample_rate * start_time)
    end_frame = int(sample_rate * end_time)

    return array[start_frame:end_frame]


def pad_or_trim(array, length: int = N_SAMPLES, *, axis: int = -1):
    """
    Pad or trim the audio array to N_SAMPLES, as expected by the encoder.

    Args:
        array: The audio array.
        length: The length to pad or trim to.
        axis: The axis to pad or trim.
    """
    if torch.is_tensor(array):
        if array.shape[axis] > length:
            array = array.index_select(
                dim=axis, index=torch.arange(length, device=array.device)
            )

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = F.pad(array, [pad for sizes in pad_widths[::-1] for pad in sizes])
    else:
        if array.shape[axis] > length:
            array = array.take(indices=range(length), axis=axis)

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = np.pad(array, pad_widths)

    return array


def get_mels_from_video_path(
    device: torch.device,
    audio_path: Path,
    n_mels: int,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
) -> torch.Tensor:
    """
    Get the Mel spectrogram from a video file.

    Args:
        device (torch.device): The device to use.
        audio_path (Path): Path to the video file.
        n_mels (int): The number of Mel filters.
        start_frame (Optional[int]): The start frame.
        end_frame (Optional[int]): The end frame.

    Returns:
        torch.Tensor: The Mel spectrogram of shape (n_mels, n_frames).
    """
    with torch.no_grad():
        audio = load_audio_from_video(audio_path)
        if start_frame is not None:
            audio = audio[start_frame:]
        if end_frame is not None:
            audio = audio[:end_frame]
        audio = pad_or_trim(audio.flatten())
        mels = log_mel_spectrogram(audio, device=device, n_mels=n_mels)
    return mels


@lru_cache(maxsize=None)
def mel_filters(device: torch.device, n_mels: int) -> torch.Tensor:
    """
    load the mel filterbank matrix for projecting STFT into a Mel spectrogram.
    Allows decoupling librosa dependency; saved using:

        np.savez_compressed(
            "mel_filters.npz",
            mel_80=librosa.filters.mel(sr=16000, n_fft=400, n_mels=80),
            mel_128=librosa.filters.mel(sr=16000, n_fft=400, n_mels=128),
        )

    Args:
        n_mels (int): The number of Mel filters to use.

    Returns:
        torch.Tensor: The mel filterbank matrix.
    """
    assert n_mels in {80, 128}, f"Unsupported n_mels: {n_mels}"

    with np.load(MEL_FILTER_PATH, allow_pickle=False) as f:
        return torch.from_numpy(f[f"mel_{n_mels}"]).to(device)


def log_mel_spectrogram(
    audio: Union[str, np.ndarray, torch.Tensor],
    n_mels: int,
    padding: int = 0,
    device: Optional[torch.device] = None,
):
    """
    Compute the log-Mel spectrogram of

    Parameters
    ----------
    audio: Union[str, np.ndarray, torch.Tensor], shape = (*)
        The path to audio or either a NumPy array or Tensor containing the audio waveform in 16 kHz

    n_mels: int
        The number of Mel-frequency filters, only 80 is supported

    padding: int
        Number of zero samples to pad to the right

    device: Optional[torch.device]
        If given, the audio tensor is moved to this device before STFT

    Returns
    -------
    torch.Tensor, shape = (80, n_frames)
        A Tensor that contains the Mel spectrogram
    """
    if not torch.is_tensor(audio):
        if isinstance(audio, str):
            audio = load_audio_from_video(audio)
        audio = torch.from_numpy(audio)

    if device is not None:
        audio = audio.to(device)
    if padding > 0:
        audio = F.pad(audio, (0, padding))
    window = torch.hann_window(N_FFT).to(audio.device)
    stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2

    filters = mel_filters(audio.device, n_mels)
    mel_spec = filters @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec


class AudioStream:
    """
    Class to mimic streaming audio input.
    """

    def __init__(
        self, audio: torch.Tensor, min_samples_per_step: int, max_samples_per_step: int
    ):
        self.audio = audio
        self.min_samples_per_step = min_samples_per_step
        self.max_samples_per_step = max_samples_per_step
        self.current_idx = 0
        self.can_step = True

    def step(self) -> torch.Tensor:
        if not self.can_step:
            raise StopIteration("End of audio stream")
        start_idx = self.current_idx
        if self.min_samples_per_step == self.max_samples_per_step:
            samples_per_step = self.min_samples_per_step
        else:
            samples_per_step = torch.randint(
                self.min_samples_per_step, self.max_samples_per_step, (1,)
            ).item()
        end_idx = min(start_idx + samples_per_step, len(self.audio))
        audio_chunk = self.audio[start_idx:end_idx]
        self.current_idx = end_idx
        if end_idx >= len(self.audio):
            self.can_step = False
        return audio_chunk
