"""Tests for tts_vc — all run without loading the Chatterbox model."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from tts_vc.audio import apply_speed, normalize_loudness, save_wav
from tts_vc.cli import build_parser, load_text, validate_args
from tts_vc.engine import DEFAULTS, WHISPER_OVERRIDES, resolve_params


# ---------------------------------------------------------------------------
# load_text
# ---------------------------------------------------------------------------

def test_load_text_inline():
    assert load_text("Hello world") == "Hello world"


def test_load_text_from_file(tmp_path):
    f = tmp_path / "speech.txt"
    f.write_text("  Hello from file  \n")
    assert load_text(str(f)) == "Hello from file"


def test_load_text_nonexistent_txt_is_inline():
    # A .txt path that doesn't exist → treated as inline string
    assert load_text("/does/not/exist.txt") == "/does/not/exist.txt"


def test_load_text_empty_file_exits(tmp_path):
    f = tmp_path / "empty.txt"
    f.write_text("   \n")
    with pytest.raises(SystemExit):
        load_text(str(f))


# ---------------------------------------------------------------------------
# argparse defaults
# ---------------------------------------------------------------------------

def test_parser_defaults(tmp_path):
    voice = tmp_path / "v.wav"
    voice.write_bytes(b"x")
    parser = build_parser()
    args = parser.parse_args([str(voice), "Hello"])
    assert args.exaggeration == 0.5
    assert args.cfg_weight == 0.5
    assert args.temperature == 0.8
    assert args.speed == 1.0
    assert args.whisper is False
    assert args.device == "auto"
    assert args.min_p == 0.05
    assert args.top_p == 1.0
    assert args.repetition_penalty == 1.2
    assert args.no_normalize is False
    assert str(args.output) == "output.wav"


def test_parser_short_flags(tmp_path):
    voice = tmp_path / "v.wav"
    voice.write_bytes(b"x")
    parser = build_parser()
    args = parser.parse_args([str(voice), "Hello", "-e", "0.9", "-c", "0.3", "-t", "1.2",
                               "-s", "1.5", "-w", "-o", "out.wav"])
    assert args.exaggeration == 0.9
    assert args.cfg_weight == 0.3
    assert args.temperature == 1.2
    assert args.speed == 1.5
    assert args.whisper is True
    assert str(args.output) == "out.wav"


def test_parser_device_choices(tmp_path):
    voice = tmp_path / "v.wav"
    voice.write_bytes(b"x")
    parser = build_parser()
    for device in ("cpu", "cuda", "mps", "auto"):
        args = parser.parse_args([str(voice), "Hello", "--device", device])
        assert args.device == device


# ---------------------------------------------------------------------------
# validate_args
# ---------------------------------------------------------------------------

def test_validate_missing_source_voice_exits(tmp_path):
    parser = build_parser()
    args = parser.parse_args([str(tmp_path / "nonexistent.wav"), "Hello"])
    with pytest.raises(SystemExit):
        validate_args(args, parser)


def test_validate_non_wav_exits(tmp_path):
    mp3 = tmp_path / "voice.mp3"
    mp3.write_bytes(b"x")
    parser = build_parser()
    args = parser.parse_args([str(mp3), "Hello"])
    with pytest.raises(SystemExit):
        validate_args(args, parser)


def test_validate_exaggeration_out_of_range_exits(tmp_path):
    voice = tmp_path / "v.wav"
    voice.write_bytes(b"x")
    parser = build_parser()
    args = parser.parse_args([str(voice), "Hello", "-e", "0.1"])  # below 0.25
    with pytest.raises(SystemExit):
        validate_args(args, parser)


def test_validate_cfg_weight_out_of_range_exits(tmp_path):
    voice = tmp_path / "v.wav"
    voice.write_bytes(b"x")
    parser = build_parser()
    args = parser.parse_args([str(voice), "Hello", "-c", "1.5"])
    with pytest.raises(SystemExit):
        validate_args(args, parser)


def test_validate_speed_out_of_range_exits(tmp_path):
    voice = tmp_path / "v.wav"
    voice.write_bytes(b"x")
    parser = build_parser()
    args = parser.parse_args([str(voice), "Hello", "-s", "3.0"])
    with pytest.raises(SystemExit):
        validate_args(args, parser)


# ---------------------------------------------------------------------------
# resolve_params (whisper mode)
# ---------------------------------------------------------------------------

def test_whisper_overrides_defaults():
    params = resolve_params(
        exaggeration=DEFAULTS["exaggeration"],
        cfg_weight=DEFAULTS["cfg_weight"],
        temperature=DEFAULTS["temperature"],
        whisper=True,
    )
    assert params["exaggeration"] == WHISPER_OVERRIDES["exaggeration"]
    assert params["cfg_weight"] == WHISPER_OVERRIDES["cfg_weight"]
    assert params["temperature"] == WHISPER_OVERRIDES["temperature"]


def test_whisper_does_not_override_explicit_values():
    # User explicitly passes --exaggeration 0.9 alongside --whisper
    params = resolve_params(exaggeration=0.9, cfg_weight=0.5, temperature=0.8, whisper=True)
    assert params["exaggeration"] == 0.9  # explicit value preserved
    assert params["cfg_weight"] == WHISPER_OVERRIDES["cfg_weight"]
    assert params["temperature"] == WHISPER_OVERRIDES["temperature"]


def test_no_whisper_returns_given_values():
    params = resolve_params(exaggeration=0.7, cfg_weight=0.3, temperature=0.6, whisper=False)
    assert params == {"exaggeration": 0.7, "cfg_weight": 0.3, "temperature": 0.6}


# ---------------------------------------------------------------------------
# audio.apply_speed
# ---------------------------------------------------------------------------

def test_apply_speed_noop_at_1():
    wav = torch.randn(1, 22050)
    out = apply_speed(wav, 22050, 1.0)
    assert out.shape == wav.shape


def test_apply_speed_2x_shortens():
    wav = torch.randn(1, 22050)
    out = apply_speed(wav, 22050, 2.0)
    assert out.shape[1] < wav.shape[1]


def test_apply_speed_half_lengthens():
    wav = torch.randn(1, 22050)
    out = apply_speed(wav, 22050, 0.5)
    assert out.shape[1] > wav.shape[1]


# ---------------------------------------------------------------------------
# audio.normalize_loudness
# ---------------------------------------------------------------------------

def test_normalize_loudness_preserves_shape():
    wav = torch.randn(1, 22050) * 0.01  # very quiet
    out = normalize_loudness(wav)
    assert out.shape == wav.shape


def test_normalize_loudness_silent_input_unchanged():
    wav = torch.zeros(1, 22050)
    out = normalize_loudness(wav)
    assert out.shape == wav.shape
    assert torch.allclose(out, wav)


def test_normalize_loudness_increases_quiet_signal():
    wav = torch.randn(1, 22050) * 0.001  # very quiet
    out = normalize_loudness(wav, target_db=-20.0)
    assert out.abs().mean() > wav.abs().mean()


# ---------------------------------------------------------------------------
# audio.save_wav
# ---------------------------------------------------------------------------

def test_save_wav_creates_file(tmp_path):
    wav = torch.randn(1, 22050)
    path = str(tmp_path / "out.wav")
    save_wav(wav, 22050, path)
    assert Path(path).exists()
    assert Path(path).stat().st_size > 0


def test_save_wav_1d_tensor(tmp_path):
    wav = torch.randn(22050)  # 1D
    path = str(tmp_path / "out_1d.wav")
    save_wav(wav, 22050, path)
    assert Path(path).exists()


# ---------------------------------------------------------------------------
# Integration test (model mocked at engine level)
# ---------------------------------------------------------------------------

def test_main_produces_output_file(tmp_path):
    voice = tmp_path / "voice.wav"
    text_file = tmp_path / "text.txt"
    output = tmp_path / "out.wav"
    voice.write_bytes(b"RIFF" + b"\x00" * 36)
    text_file.write_text("Hello world")

    fake_wav = torch.randn(1, 22050)
    fake_model = MagicMock()
    fake_model.generate.return_value = fake_wav
    fake_model.sr = 22050

    with patch("tts_vc.engine.ChatterboxTTS") as mock_cls:
        mock_cls.from_pretrained.return_value = fake_model
        with patch("sys.argv", ["tts", str(voice), str(text_file), "-o", str(output)]):
            from tts_vc.cli import main
            main()

    assert output.exists()
    assert output.stat().st_size > 0


def test_main_whisper_mode(tmp_path):
    voice = tmp_path / "voice.wav"
    voice.write_bytes(b"RIFF" + b"\x00" * 36)
    output = tmp_path / "out.wav"

    fake_wav = torch.randn(1, 22050)
    fake_model = MagicMock()
    fake_model.generate.return_value = fake_wav
    fake_model.sr = 22050

    with patch("tts_vc.engine.ChatterboxTTS") as mock_cls:
        mock_cls.from_pretrained.return_value = fake_model
        with patch("sys.argv", ["tts", str(voice), "Hello", "--whisper", "-o", str(output)]):
            from tts_vc.cli import main
            main()

    # Verify whisper params were passed to generate()
    call_kwargs = fake_model.generate.call_args.kwargs
    assert call_kwargs["exaggeration"] == WHISPER_OVERRIDES["exaggeration"]
    assert call_kwargs["cfg_weight"] == WHISPER_OVERRIDES["cfg_weight"]
    assert call_kwargs["temperature"] == WHISPER_OVERRIDES["temperature"]
