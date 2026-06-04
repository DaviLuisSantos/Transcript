# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A single-file CLI tool (`transcribe.py`) that downloads audio from YouTube videos via `yt-dlp` and transcribes them locally using `faster-whisper` (CTranslate2 backend).

## Environment Setup

```powershell
# Activate the virtual environment (Windows)
.venv\Scripts\Activate.ps1

# Install / sync dependencies
pip install faster-whisper yt-dlp
```

`ffmpeg` must be installed separately (e.g. `winget install ffmpeg`). The script auto-discovers winget installs; no manual PATH change needed.

## Running the Script

```powershell
# YouTube URL (output named after video title)
python transcribe.py <YouTube_URL>

# Local video/audio file (output named after file)
python transcribe.py video.mp4

# With options
python transcribe.py <source> --model large-v2 --language en --threads 8 --fast
```

| Flag | Default | Notes |
|---|---|---|
| `--model` / `-m` | `medium` | `tiny` → `large-v3`; larger = more accurate, slower |
| `--language` / `-l` | `pt` | BCP-47 language code |
| `--output` / `-o` | video title / filename | Output `.txt` path |
| `--threads` / `-t` | all logical cores | CPU thread count |
| `--fast` / `-f` | off | `beam_size=1` greedy mode, ~2× faster |

## Architecture

All logic lives in `transcribe.py`. Execution flow:

1. **`_ensure_ffmpeg_in_path()`** — runs at module import; patches `PATH` for winget ffmpeg installs before any subprocess calls.
2. **`is_local_file()`** — determines whether `source` is a local path or a URL; drives the branch in `main()`.
3. **`check_dependencies(need_ytdlp)`** — verifies `faster_whisper` is importable and, when `need_ytdlp=True` (URL mode), that `yt-dlp` is callable. Also checks `ffmpeg` is in PATH.
4. **`download_audio()`** — shells out to `yt-dlp` with `--extract-audio --audio-format wav`; writes to a `tempfile.TemporaryDirectory` that is cleaned up automatically after transcription.
5. **`transcribe()`** — loads `WhisperModel`, calls `model.transcribe()` with VAD filtering enabled, consumes the lazy segment generator through a `tqdm` progress bar.
6. **`save_transcript()`** — writes UTF-8 `.txt`.

**Device detection**: `detect_device()` tries `torch.cuda` first; falls back to CPU/INT8. CUDA is used only if `torch` is installed in the venv.

**Output naming**: when no `--output` is given, `get_video_title()` calls `yt-dlp --get-title` and sanitises Windows-illegal characters, truncating to 80 chars.

## Key Constraints

- The script forces UTF-8 on Windows stdout/stderr at import time — keep all print statements emoji-safe or use ASCII fallbacks.
- `yt-dlp` is resolved relative to `sys.executable`'s Scripts dir first, so it works correctly inside the venv without relying on a system-wide install.
- The `tempfile.TemporaryDirectory` context manager in `main()` means audio files are never persisted; transcription must complete before the context exits.
