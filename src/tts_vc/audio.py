"""Audio post-processing: speed adjustment, loudness normalization, WAV save."""

import math

import torch
import torchaudio


def apply_speed(wav: torch.Tensor, sample_rate: int, speed: float) -> torch.Tensor:
    """Adjust playback speed via resampling (no pitch shift).

    Resamples the tensor to a virtual rate that, when played back at
    the original sample_rate, produces faster or slower speech.
    """
    if abs(speed - 1.0) < 1e-4:
        return wav
    # Treat the audio as if it were recorded at (sample_rate * speed) Hz,
    # then resample back to sample_rate — this speeds up or slows down
    # playback without altering pitch.
    new_sr = int(sample_rate * speed)
    return torchaudio.functional.resample(wav, orig_freq=new_sr, new_freq=sample_rate)


def normalize_loudness(wav: torch.Tensor, target_db: float = -20.0) -> torch.Tensor:
    """RMS-normalize the waveform to target_db.

    This is especially important for whisper mode, where the model output
    can be very quiet and otherwise inaudible.
    """
    rms = wav.pow(2).mean().sqrt()
    if rms < 1e-8:
        # Silent or near-silent — return unchanged to avoid amplifying noise.
        return wav
    target_rms = 10 ** (target_db / 20.0)
    return wav * (target_rms / rms)


def save_wav(wav: torch.Tensor, sample_rate: int, path: str) -> None:
    """Save a waveform tensor to a WAV file."""
    # torchaudio expects shape (channels, samples); add channel dim if needed.
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    torchaudio.save(path, wav.cpu(), sample_rate)
