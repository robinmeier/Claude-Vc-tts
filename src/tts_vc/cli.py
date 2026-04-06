"""Voice cloning TTS CLI.

Usage:
    uv run tts sourcevoice.wav speech.txt
    uv run tts sourcevoice.wav "Hello world" -e 0.8 -o out.wav
    uv run tts sourcevoice.wav speech.txt --whisper
    uv run tts --list-tags
"""

import argparse
import os
import sys


PARALINGUISTIC_TAGS = {
    "laugh": "[laugh]",
    "chuckle": "[chuckle]",
    "sigh": "[sigh]",
    "cough": "[cough]",
    "whisper": "[whisper]",
    "clear_throat": "[clears throat]",
    "breath": "[breath]",
    "hesitation": "[hesitation]",
}

# Whisper mode parameter presets (also prepends [whisper] tag)
WHISPER_EXAGGERATION = 0.1
WHISPER_CFG_WEIGHT = 0.2


def auto_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def load_text(text_arg: str) -> str:
    """Load text from a file path or return as-is if it's an inline string."""
    if text_arg.endswith(".txt") and os.path.isfile(text_arg):
        with open(text_arg, "r", encoding="utf-8") as f:
            return f.read().strip()
    return text_arg


def build_text(text: str, tags: list[str] | None) -> str:
    """Append any requested paralinguistic tags to the text."""
    if not tags:
        return text
    tag_strs = []
    for tag in tags:
        if tag in PARALINGUISTIC_TAGS:
            tag_strs.append(PARALINGUISTIC_TAGS[tag])
        else:
            print(f"Warning: unknown tag '{tag}', ignoring. Use --list-tags to see valid tags.", file=sys.stderr)
    return text + " " + " ".join(tag_strs) if tag_strs else text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tts",
        description="Clone a voice and synthesize speech with style control.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run tts voice.wav speech.txt
  uv run tts voice.wav "Hello world" -e 0.9 -o out.wav
  uv run tts voice.wav speech.txt --whisper
  uv run tts voice.wav "That's funny [laugh]" --tags laugh
  uv run tts --list-tags
        """,
    )

    parser.add_argument(
        "--list-tags",
        action="store_true",
        help="Print available paralinguistic tags and exit.",
    )

    parser.add_argument(
        "source_voice",
        nargs="?",
        help="Path to reference WAV file (voice to clone).",
    )
    parser.add_argument(
        "text",
        nargs="?",
        help="Text to synthesize: path to .txt file or inline string.",
    )

    parser.add_argument(
        "-o", "--output",
        default="output.wav",
        help="Output WAV file path (default: output.wav).",
    )
    parser.add_argument(
        "-e", "--exaggeration",
        type=float,
        default=0.5,
        metavar="FLOAT",
        help="Emotional expressiveness: 0.0 (flat), 0.5 (normal), 1.0+ (exaggerated, max ~2.0). Default: 0.5.",
    )
    parser.add_argument(
        "-t", "--temperature",
        type=float,
        default=0.8,
        metavar="FLOAT",
        help="Voice randomness: 0.5-0.7 (consistent), 1.0-1.5 (expressive). Default: 0.8.",
    )
    parser.add_argument(
        "-c", "--cfg-weight",
        type=float,
        default=0.5,
        metavar="FLOAT",
        help="CFG guidance weight 0.0 (variable) to 1.0 (controlled). Default: 0.5.",
    )
    parser.add_argument(
        "-s", "--speed",
        type=float,
        default=1.0,
        metavar="FLOAT",
        help="Speaking speed multiplier (default: 1.0). Not yet supported by Chatterbox.",
    )
    parser.add_argument(
        "-w", "--whisper",
        action="store_true",
        help=f"Whisper mode: prepends [whisper] tag and sets exaggeration={WHISPER_EXAGGERATION}, cfg-weight={WHISPER_CFG_WEIGHT}.",
    )
    parser.add_argument(
        "-d", "--device",
        default=None,
        choices=["cpu", "cuda", "mps"],
        help="Device to run on (default: auto-detect).",
    )
    parser.add_argument(
        "--tags",
        nargs="+",
        metavar="TAG",
        help="Paralinguistic tags to append to text (e.g. --tags laugh sigh). Use --list-tags for options.",
    )

    return parser


def generate(
    source_voice: str,
    text: str,
    output: str,
    exaggeration: float,
    cfg_weight: float,
    temperature: float,
    device: str,
) -> None:
    try:
        from chatterbox.tts import ChatterboxTTS
        import torchaudio
    except ImportError:
        print(
            "Error: chatterbox-tts is not installed.\n"
            "Run: uv sync  (or pip install chatterbox-tts torchaudio)",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Loading model on {device}... (first run downloads ~1-2 GB from HuggingFace)", file=sys.stderr)
    model = ChatterboxTTS.from_pretrained(device=device)

    print("Generating speech...", file=sys.stderr)
    wav = model.generate(
        text=text,
        audio_prompt_path=source_voice,
        exaggeration=exaggeration,
        cfg_weight=cfg_weight,
        temperature=temperature,
    )

    torchaudio.save(output, wav, model.sr)
    print(f"Saved: {output}", file=sys.stderr)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.list_tags:
        print("Available paralinguistic tags (use with --tags or inline in text):")
        for name, tag in PARALINGUISTIC_TAGS.items():
            print(f"  {name:<15} -> {tag}")
        return

    if not args.source_voice or not args.text:
        parser.error("source_voice and text are required (unless using --list-tags).")

    if not os.path.isfile(args.source_voice):
        parser.error(f"Source voice file not found: {args.source_voice}")

    exaggeration = args.exaggeration
    cfg_weight = args.cfg_weight

    if args.whisper:
        exaggeration = WHISPER_EXAGGERATION
        cfg_weight = WHISPER_CFG_WEIGHT
        print(
            f"Whisper mode: exaggeration={exaggeration}, cfg_weight={cfg_weight}, prepending [whisper] tag",
            file=sys.stderr,
        )

    if exaggeration < 0.0:
        parser.error("--exaggeration must be >= 0.0")
    if not 0.0 <= cfg_weight <= 1.0:
        parser.error("--cfg-weight must be between 0.0 and 1.0")
    if not 0.05 <= args.temperature <= 2.0:
        parser.error("--temperature must be between 0.05 and 2.0")
    if args.speed != 1.0:
        print("Warning: --speed is not yet implemented by Chatterbox; ignored.", file=sys.stderr)

    device = args.device or auto_device()
    text = load_text(args.text)
    if args.whisper:
        text = f"[whisper] {text}"
    text = build_text(text, args.tags)

    generate(
        source_voice=args.source_voice,
        text=text,
        output=args.output,
        exaggeration=exaggeration,
        cfg_weight=cfg_weight,
        temperature=args.temperature,
        device=device,
    )


if __name__ == "__main__":
    main()
