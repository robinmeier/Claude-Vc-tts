# tts-vc — Voice Cloning TTS

Clone any voice from a short WAV sample and synthesize speech with control over
emotion, pacing, and style. Built on [Chatterbox TTS](https://github.com/resemble-ai/chatterbox)
(Resemble AI, MIT licensed).

## Requirements

- **macOS** (Apple Silicon or Intel) — MPS acceleration auto-detected on Apple Silicon
- **Python 3.10+**
- **[uv](https://docs.astral.sh/uv/getting-started/installation/)** — fast Python package manager

## Installation

```bash
# 1. Clone the repo
git clone https://github.com/robinmeier/Claude-Vc-tts.git
cd Claude-Vc-tts

# 2. Check out the working branch
git checkout claude/voice-cloning-tts-research-lbiXt

# 3. Install dependencies (creates a .venv automatically)
uv sync

# 4. Verify
uv run tts --help
```

> **First run:** Chatterbox model weights (~2 GB) are downloaded from HuggingFace
> automatically on first use and cached in `~/.cache/huggingface/`.

## Quick Start

```bash
# Clone a voice and speak a text file
uv run tts myvoice.wav speech.txt

# Inline text
uv run tts myvoice.wav "Hello, this is a cloned voice."

# Save to a specific file
uv run tts myvoice.wav speech.txt -o result.wav
```

## Options

```
positional arguments:
  source_voice          Reference WAV file for voice cloning (10+ sec recommended)
  text                  Text to synthesize: path to .txt file or inline string

output:
  -o, --output FILE     Output WAV file (default: output.wav)

style control:
  -e, --exaggeration    Emotion intensity: 0.25 (flat) → 0.5 (normal) → 2.0 (exaggerated)  [default: 0.5]
  -c, --cfg-weight      CFG guidance 0.0–1.0: lower = slower/softer pacing                 [default: 0.5]
  -t, --temperature     Sampling temperature 0.05–5.0: lower = more consistent voice        [default: 0.8]
  -s, --speed           Playback speed multiplier 0.5–2.0                                   [default: 1.0]
  -w, --whisper         Whisper mode (lowers exaggeration, cfg-weight, and temperature)
  --tags TAG [TAG ...]  Append paralinguistic tags to text (see --list-tags)

model:
  --model               standard (English, default) or multilingual (23 languages)
  -l, --language CODE   Language for --model multilingual (default: en)

advanced:
  --device              cpu / cuda / mps / auto (default: auto-detect)
  --min-p               Min-p sampling threshold          [default: 0.05]
  --top-p               Top-p nucleus sampling            [default: 1.0]
  --repetition-penalty  Repetition penalty                [default: 1.2]
  --no-normalize        Skip loudness normalization of output
  --list-tags           Print available paralinguistic tags and exit
```

## Examples

### Basic voice cloning
```bash
uv run tts speaker.wav "I cannot believe it happened."
```

### High emotion
```bash
uv run tts speaker.wav "This is incredible!" -e 1.5 -c 0.3 -o excited.wav
```

### Whisper mode
```bash
uv run tts speaker.wav "Can you keep a secret?" --whisper -o secret.wav
```

### Adjust speed
```bash
uv run tts speaker.wav speech.txt -s 0.85 -o slow.wav
```

### Paralinguistic tags
```bash
# Add a laugh at the end
uv run tts speaker.wav "That's the funniest thing I've heard" --tags laugh

# See all available tags
uv run tts --list-tags
```

Available tags: `laugh` `chuckle` `sigh` `cough` `whisper` `clear_throat` `breath` `hesitation`

Tags can also be embedded directly in text: `"That was hilarious [laugh] I can't stop"`.

### Multilingual (23 languages)
```bash
# French
uv run tts speaker.wav "Bonjour, comment allez-vous ?" --model multilingual -l fr

# Japanese
uv run tts speaker.wav "こんにちは、元気ですか？" --model multilingual -l ja

# German
uv run tts speaker.wav "Guten Morgen, wie geht es Ihnen?" --model multilingual -l de
```

Supported language codes: `ar` `da` `de` `el` `en` `es` `fi` `fr` `he` `hi` `it` `ja` `ko` `ms` `nl` `no` `pl` `pt` `ru` `sv` `sw` `tr` `zh`

> The multilingual model (~500M parameters) downloads separately on first use.

## Tips for Best Results

| Goal | Recommended settings |
|------|---------------------|
| Natural conversational speech | defaults (`-e 0.5 -c 0.5`) |
| Flat, calm narration | `-e 0.3 -c 0.7` |
| Expressive / emotional | `-e 1.0 -c 0.3 -t 1.0` |
| Whispered speech | `--whisper` |
| Slow, deliberate pacing | `-c 0.2 -s 0.85` |
| Fast-paced speech | `-e 0.7 -s 1.2` |

**Reference audio:** 10–30 seconds of clean, single-speaker audio gives the best
voice cloning. Noisy or very short clips reduce quality.

## Apple Silicon notes

Device is auto-detected (`mps` on Apple Silicon, `cpu` on Intel Mac).
If you hit an unsupported MPS operation, prefix your command with:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run tts ...
```

## Running Tests

```bash
uv run pytest tests/ -v
```

Tests are fast and run without downloading any model (model is mocked).

## Project Structure

```
src/tts_vc/
├── cli.py      # argument parsing, entry point
├── engine.py   # model loading, device detection, speech generation
└── audio.py    # speed adjustment, loudness normalization, WAV save
tests/
└── test_cli.py
pyproject.toml
```

## License

Code: MIT. Chatterbox model weights: see [Chatterbox repository](https://github.com/resemble-ai/chatterbox) for terms.
