"""Model loading, device detection, and speech generation."""

import sys
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from chatterbox.tts import ChatterboxTTS

DEFAULTS = {
    "exaggeration": 0.5,
    "cfg_weight": 0.5,
    "temperature": 0.8,
}

WHISPER_OVERRIDES = {
    "exaggeration": 0.2,
    "cfg_weight": 0.2,
    "temperature": 0.4,
}


def detect_device() -> str:
    """Return the best available device: mps, cuda, or cpu."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _patch_perth() -> None:
    """Patch perth.PerthImplicitWatermarker with a no-op if it failed to load.

    On Mac (MPS/CPU), the resemble-perth native extension often fails to build,
    leaving PerthImplicitWatermarker set to None. Chatterbox still works fine
    without watermarking — we just skip it silently.
    """
    try:
        import perth
        if perth.PerthImplicitWatermarker is None:
            class _NoOpWatermarker:
                def apply_watermark(self, wav, sample_rate=None):
                    return wav
            perth.PerthImplicitWatermarker = _NoOpWatermarker
    except ImportError:
        pass


def load_model(device: str) -> "ChatterboxTTS":
    """Load ChatterboxTTS from HuggingFace (downloads ~2GB on first run)."""
    _patch_perth()
    try:
        from chatterbox.tts import ChatterboxTTS
    except ImportError:
        print(
            "Error: chatterbox-tts is not installed.\n"
            "Run: uv sync  (or: pip install chatterbox-tts torchaudio)",
            file=sys.stderr,
        )
        sys.exit(1)
    return ChatterboxTTS.from_pretrained(device=device)


def resolve_params(
    exaggeration: float,
    cfg_weight: float,
    temperature: float,
    whisper: bool,
) -> dict:
    """Return generation params, applying whisper overrides only where the user
    left values at their defaults (explicit user values always win)."""
    params = {
        "exaggeration": exaggeration,
        "cfg_weight": cfg_weight,
        "temperature": temperature,
    }
    if whisper:
        for key, override in WHISPER_OVERRIDES.items():
            if params[key] == DEFAULTS[key]:
                params[key] = override
    return params


def synthesize(
    model: "ChatterboxTTS",
    text: str,
    audio_prompt_path: str,
    exaggeration: float,
    cfg_weight: float,
    temperature: float,
    min_p: float = 0.05,
    top_p: float = 1.0,
    repetition_penalty: float = 1.2,
) -> tuple[torch.Tensor, int]:
    """Generate speech and return (wav_tensor, sample_rate)."""
    wav = model.generate(
        text,
        audio_prompt_path=audio_prompt_path,
        exaggeration=exaggeration,
        cfg_weight=cfg_weight,
        temperature=temperature,
        min_p=min_p,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )
    return wav, model.sr
