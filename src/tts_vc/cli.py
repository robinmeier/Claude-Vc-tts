"""Voice cloning TTS CLI.

Usage:
    uv run tts sourcevoice.wav speech.txt
    uv run tts sourcevoice.wav "Hello world" -e 0.8 -o out.wav
    uv run tts sourcevoice.wav speech.txt --whisper
    uv run tts sourcevoice.wav "Bonjour le monde" --model multilingual -l fr
    uv run tts --list-tags
"""

import sys
from pathlib import Path

from tts_vc.engine import MULTILINGUAL_LANGUAGES

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


def build_text(text: str, tags: list[str] | None) -> str:
    """Append requested paralinguistic tags to the text."""
    if not tags:
        return text
    tag_strs = []
    for tag in tags:
        if tag in PARALINGUISTIC_TAGS:
            tag_strs.append(PARALINGUISTIC_TAGS[tag])
        else:
            print(
                f"Warning: unknown tag '{tag}', ignoring. Run --list-tags to see valid options.",
                file=sys.stderr,
            )
    return (text + " " + " ".join(tag_strs)).rstrip() if tag_strs else text


def build_parser():
    import argparse

    parser = argparse.ArgumentParser(
        prog="tts",
        description="Clone a voice and synthesize speech with style control.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run tts voice.wav speech.txt
  uv run tts voice.wav "Hello world" -e 0.9 -o out.wav
  uv run tts voice.wav speech.txt --whisper -o whisper.wav
  uv run tts voice.wav "That's funny" --tags laugh -o funny.wav
  uv run tts voice.wav "Bonjour le monde" --model multilingual -l fr
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
        type=Path,
        nargs="?",
        help="Reference WAV file for voice cloning (10+ seconds recommended).",
    )
    parser.add_argument(
        "text",
        nargs="?",
        help="Text to synthesize: path to .txt file or inline string.",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("output.wav"),
        help="Output WAV file path (default: output.wav).",
    )
    parser.add_argument(
        "-e", "--exaggeration",
        type=float,
        default=0.5,
        metavar="FLOAT",
        help="Emotion intensity: 0.25 (flat) → 0.5 (normal) → 2.0 (exaggerated). Default: 0.5.",
    )
    parser.add_argument(
        "-c", "--cfg-weight",
        type=float,
        default=0.5,
        metavar="FLOAT",
        help="CFG guidance 0.0–1.0: lower = slower/softer pacing. Default: 0.5.",
    )
    parser.add_argument(
        "-t", "--temperature",
        type=float,
        default=0.8,
        metavar="FLOAT",
        help="Sampling temperature 0.05–5.0: lower = more consistent voice. Default: 0.8.",
    )
    parser.add_argument(
        "-s", "--speed",
        type=float,
        default=1.0,
        metavar="FLOAT",
        help="Playback speed multiplier 0.5–2.0 (default: 1.0).",
    )
    parser.add_argument(
        "-w", "--whisper",
        action="store_true",
        help="Whisper mode: sets soft defaults for exaggeration, cfg-weight, and temperature.",
    )
    parser.add_argument(
        "--model",
        choices=["standard", "multilingual"],
        default="standard",
        help="Model to use: standard (English, default) or multilingual (23 languages).",
    )
    parser.add_argument(
        "-l", "--language",
        default="en",
        metavar="CODE",
        help=(
            "Language code for --model multilingual (default: en). "
            f"Supported: {', '.join(sorted(MULTILINGUAL_LANGUAGES))}."
        ),
    )
    parser.add_argument(
        "--tags",
        nargs="+",
        metavar="TAG",
        help="Append paralinguistic tags to text (e.g. --tags laugh sigh). See --list-tags.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "mps", "auto"],
        default="auto",
        help="Compute device (default: auto-detect).",
    )
    parser.add_argument(
        "--min-p",
        type=float,
        default=0.05,
        help="Min-p sampling threshold (default: 0.05).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p nucleus sampling (default: 1.0).",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.2,
        help="Repetition penalty (default: 1.2).",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Skip loudness normalization of output audio.",
    )

    return parser


def validate_args(args, parser) -> None:
    if args.list_tags:
        return
    if not args.source_voice or not args.text:
        parser.error("source_voice and text are required (or use --list-tags).")
    if not args.source_voice.exists():
        parser.error(f"Source voice file not found: {args.source_voice}")
    if args.source_voice.suffix.lower() != ".wav":
        parser.error(f"source_voice must be a .wav file, got: {args.source_voice}")
    if not (0.25 <= args.exaggeration <= 2.0):
        parser.error("--exaggeration must be between 0.25 and 2.0")
    if not (0.0 <= args.cfg_weight <= 1.0):
        parser.error("--cfg-weight must be between 0.0 and 1.0")
    if not (0.05 <= args.temperature <= 5.0):
        parser.error("--temperature must be between 0.05 and 5.0")
    if not (0.5 <= args.speed <= 2.0):
        parser.error("--speed must be between 0.5 and 2.0")
    if args.model == "multilingual" and args.language not in MULTILINGUAL_LANGUAGES:
        parser.error(
            f"Unknown language code '{args.language}'. "
            f"Supported: {', '.join(sorted(MULTILINGUAL_LANGUAGES))}."
        )


def load_text(text_arg: str) -> str:
    """Return text from file if arg is an existing .txt path, else use as-is."""
    p = Path(text_arg)
    if p.suffix.lower() == ".txt" and p.exists():
        content = p.read_text(encoding="utf-8").strip()
        if not content:
            print(f"Error: text file is empty: {p}", file=sys.stderr)
            sys.exit(1)
        return content
    return text_arg.strip()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.list_tags:
        print("Available paralinguistic tags (use with --tags or inline in text):")
        for name, tag in PARALINGUISTIC_TAGS.items():
            print(f"  {name:<15} -> {tag}")
        return

    validate_args(args, parser)

    text = load_text(args.text)
    text = build_text(text, args.tags)

    # Deferred imports: keeps `uv run tts --help` instant (torch takes ~3s to import)
    from tts_vc.engine import detect_device, load_model, resolve_params, synthesize
    from tts_vc.audio import apply_speed, normalize_loudness, save_wav

    device = detect_device() if args.device == "auto" else args.device
    multilingual = args.model == "multilingual"

    model_label = f"Chatterbox {'Multilingual' if multilingual else 'Standard'}"
    print(f"Loading {model_label} model on {device}... (first run downloads ~2 GB)", file=sys.stderr)
    model = load_model(device, multilingual=multilingual)

    if multilingual:
        lang_name = MULTILINGUAL_LANGUAGES.get(args.language, args.language)
        print(f"Language: {args.language} ({lang_name})", file=sys.stderr)

    params = resolve_params(
        exaggeration=args.exaggeration,
        cfg_weight=args.cfg_weight,
        temperature=args.temperature,
        whisper=args.whisper,
    )
    if args.whisper:
        print(
            f"Whisper mode: exaggeration={params['exaggeration']}, "
            f"cfg_weight={params['cfg_weight']}, temperature={params['temperature']}",
            file=sys.stderr,
        )

    print(f"Synthesizing {len(text)} characters...", file=sys.stderr)
    wav, sr = synthesize(
        model=model,
        text=text,
        audio_prompt_path=str(args.source_voice),
        **params,
        min_p=args.min_p,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        language_id=args.language if multilingual else None,
    )

    if args.speed != 1.0:
        wav = apply_speed(wav, sr, args.speed)

    if not args.no_normalize:
        wav = normalize_loudness(wav)

    save_wav(wav, sr, str(args.output))
    print(f"Saved: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
