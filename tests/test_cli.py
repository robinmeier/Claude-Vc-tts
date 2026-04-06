"""Tests for tts_vc CLI."""

import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from tts_vc.cli import (
    PARALINGUISTIC_TAGS,
    WHISPER_CFG_WEIGHT,
    WHISPER_EXAGGERATION,
    auto_device,
    build_parser,
    build_text,
    load_text,
)


# ---------------------------------------------------------------------------
# load_text
# ---------------------------------------------------------------------------

def test_load_text_inline():
    assert load_text("Hello world") == "Hello world"


def test_load_text_from_file(tmp_path):
    f = tmp_path / "speech.txt"
    f.write_text("  Hello from file  \n")
    assert load_text(str(f)) == "Hello from file"


def test_load_text_nonexistent_txt_treated_as_inline():
    # A .txt path that doesn't exist is treated as an inline string
    result = load_text("/does/not/exist.txt")
    assert result == "/does/not/exist.txt"


# ---------------------------------------------------------------------------
# build_text
# ---------------------------------------------------------------------------

def test_build_text_no_tags():
    assert build_text("Hello", None) == "Hello"
    assert build_text("Hello", []) == "Hello"


def test_build_text_known_tag():
    result = build_text("Hello", ["laugh"])
    assert "[laugh]" in result
    assert result.startswith("Hello")


def test_build_text_multiple_tags():
    result = build_text("Hello", ["laugh", "sigh"])
    assert "[laugh]" in result
    assert "[sigh]" in result


def test_build_text_unknown_tag_warns(capsys):
    result = build_text("Hello", ["nonexistent_tag"])
    captured = capsys.readouterr()
    assert "unknown tag" in captured.err
    assert result == "Hello"


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------

def test_parser_defaults():
    parser = build_parser()
    args = parser.parse_args(["voice.wav", "Hello"])
    assert args.source_voice == "voice.wav"
    assert args.text == "Hello"
    assert args.output == "output.wav"
    assert args.exaggeration == 0.5
    assert args.cfg_weight == 0.5
    assert args.speed == 1.0
    assert args.whisper is False
    assert args.device is None
    assert args.tags is None
    assert args.list_tags is False


def test_parser_whisper_flag():
    parser = build_parser()
    args = parser.parse_args(["voice.wav", "Hello", "--whisper"])
    assert args.whisper is True


def test_parser_short_flags():
    parser = build_parser()
    args = parser.parse_args(["voice.wav", "Hello", "-e", "0.9", "-c", "0.3", "-o", "out.wav", "-w"])
    assert args.exaggeration == 0.9
    assert args.cfg_weight == 0.3
    assert args.output == "out.wav"
    assert args.whisper is True


def test_parser_device_choices():
    parser = build_parser()
    for device in ("cpu", "cuda", "mps"):
        args = parser.parse_args(["voice.wav", "Hello", "-d", device])
        assert args.device == device


def test_parser_list_tags():
    parser = build_parser()
    args = parser.parse_args(["--list-tags"])
    assert args.list_tags is True


# ---------------------------------------------------------------------------
# whisper mode overrides
# ---------------------------------------------------------------------------

def test_whisper_mode_overrides_exaggeration_and_cfg(tmp_path):
    """main() applies whisper overrides when --whisper is passed."""
    voice = tmp_path / "v.wav"
    voice.write_bytes(b"RIFF")  # dummy file so os.path.isfile passes

    mock_wav = MagicMock()
    mock_model = MagicMock()
    mock_model.generate.return_value = mock_wav
    mock_model.sr = 24000

    with patch("tts_vc.cli.generate") as mock_generate:
        sys.argv = ["tts", str(voice), "Hello", "--whisper"]
        from tts_vc.cli import main
        main()

    mock_generate.assert_called_once()
    call_kwargs = mock_generate.call_args
    assert call_kwargs.kwargs["exaggeration"] == WHISPER_EXAGGERATION
    assert call_kwargs.kwargs["cfg_weight"] == WHISPER_CFG_WEIGHT


# ---------------------------------------------------------------------------
# auto_device
# ---------------------------------------------------------------------------

def test_auto_device_cuda(monkeypatch):
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True
    monkeypatch.setitem(sys.modules, "torch", mock_torch)
    assert auto_device() == "cuda"


def test_auto_device_mps(monkeypatch):
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    mock_torch.backends.mps.is_available.return_value = True
    monkeypatch.setitem(sys.modules, "torch", mock_torch)
    assert auto_device() == "mps"


def test_auto_device_cpu(monkeypatch):
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    mock_torch.backends.mps.is_available.return_value = False
    monkeypatch.setitem(sys.modules, "torch", mock_torch)
    assert auto_device() == "cpu"


# ---------------------------------------------------------------------------
# list-tags output
# ---------------------------------------------------------------------------

def test_list_tags_output(capsys):
    sys.argv = ["tts", "--list-tags"]
    from tts_vc.cli import main
    main()
    captured = capsys.readouterr()
    for name in PARALINGUISTIC_TAGS:
        assert name in captured.out
