#!/usr/bin/env python3
"""
transcribe.py — Transcreve vídeos do YouTube ou arquivos locais usando yt-dlp + faster-whisper

Uso:
    python transcribe.py <URL ou arquivo> [opções]

Exemplos:
    python transcribe.py https://youtu.be/xxxxxxxxxxx
    python transcribe.py video.mp4
    python transcribe.py https://youtu.be/xxxxxxxxxxx --model large-v2
    python transcribe.py video.mp4 --output minha_transcricao.txt
    python transcribe.py https://youtu.be/xxxxxxxxxxx --language en
    python transcribe.py https://youtu.be/xxxxxxxxxxx --threads 8
"""

import argparse
import os
import sys
import tempfile
import subprocess

# Force UTF-8 output on Windows terminals
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Suprime avisos desnecessários do HuggingFace Hub
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import warnings
warnings.filterwarnings("ignore", message=".*unauthenticated.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*huggingface_hub.*", category=UserWarning)


def _ensure_ffmpeg_in_path():
    """Adiciona ffmpeg ao PATH se não estiver (instalações via winget no Windows)."""
    import shutil
    if shutil.which("ffmpeg"):
        return
    winget_base = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Microsoft", "WinGet", "Packages")
    if os.path.isdir(winget_base):
        for entry in os.listdir(winget_base):
            if "ffmpeg" in entry.lower():
                for root, dirs, files in os.walk(os.path.join(winget_base, entry)):
                    if "ffmpeg.exe" in files:
                        os.environ["PATH"] = root + os.pathsep + os.environ.get("PATH", "")
                        return

_ensure_ffmpeg_in_path()


def is_local_file(source: str) -> bool:
    return os.path.isfile(source)


def get_yt_dlp_cmd():
    """Retorna o caminho do executável yt-dlp (venv ou sistema)."""
    scripts_dir = os.path.dirname(sys.executable)
    for name in ("yt-dlp.exe", "yt-dlp"):
        candidate = os.path.join(scripts_dir, name)
        if os.path.isfile(candidate):
            return candidate
    return "yt-dlp"


def find_ffmpeg():
    """Procura ffmpeg no PATH e em locais comuns do winget."""
    import shutil
    if shutil.which("ffmpeg"):
        return None  # já no PATH, não precisa --ffmpeg-location

    winget_base = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Microsoft", "WinGet", "Packages")
    if os.path.isdir(winget_base):
        for entry in os.listdir(winget_base):
            if "ffmpeg" in entry.lower():
                for root, dirs, files in os.walk(os.path.join(winget_base, entry)):
                    if "ffmpeg.exe" in files:
                        return root
    return None


def detect_device():
    """Detecta o melhor dispositivo disponível (cuda > cpu)."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda", "float16"
    except ImportError:
        pass
    return "cpu", "int8"


def check_dependencies(need_ytdlp: bool = True):
    """Verifica se faster-whisper e (opcionalmente) yt-dlp estão instalados."""
    pip_missing = []
    system_missing = []

    try:
        import faster_whisper
    except ImportError:
        pip_missing.append("faster-whisper")

    if need_ytdlp:
        try:
            result = subprocess.run([get_yt_dlp_cmd(), "--version"], capture_output=True)
            if result.returncode != 0:
                pip_missing.append("yt-dlp")
        except FileNotFoundError:
            pip_missing.append("yt-dlp")

    import shutil
    if not shutil.which("ffmpeg"):
        system_missing.append("ffmpeg")

    if pip_missing:
        print("❌ Dependências Python faltando. Instale com:")
        print(f"   pip install {' '.join(pip_missing)}")

    if system_missing:
        print("❌ ffmpeg não encontrado no sistema.")
        print("   Instale via winget:  winget install ffmpeg")
        print("   Após instalar, reinicie o terminal.")

    if pip_missing or system_missing:
        sys.exit(1)


def download_audio(url: str, output_dir: str) -> str:
    """Baixa apenas o áudio do vídeo via yt-dlp."""
    print(f"⬇️  Baixando áudio de: {url}")

    output_template = os.path.join(output_dir, "audio.%(ext)s")

    cmd = [
        get_yt_dlp_cmd(),
        "--extract-audio",
        "--audio-format", "wav",   # WAV: faster-whisper lê direto, sem recodificação extra
        "--audio-quality", "0",
        "--output", output_template,
        "--no-playlist",
    ]

    ffmpeg_dir = find_ffmpeg()
    if ffmpeg_dir:
        cmd += ["--ffmpeg-location", ffmpeg_dir]

    cmd.append(url)

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("❌ Erro ao baixar o vídeo:")
        print(result.stderr)
        sys.exit(1)

    for f in os.listdir(output_dir):
        if f.startswith("audio"):
            return os.path.join(output_dir, f)

    print("❌ Arquivo de áudio não encontrado após download.")
    sys.exit(1)


def get_video_title(url: str) -> str:
    """Pega o título do vídeo para usar como nome do arquivo de saída."""
    cmd = [get_yt_dlp_cmd(), "--get-title", "--no-playlist", url]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        title = result.stdout.strip()
        invalid = r'\/:*?"<>|'
        for char in invalid:
            title = title.replace(char, "_")
        return title[:80]

    return "transcricao"


def transcribe(audio_path: str, model_name: str, language: str, threads: int, fast: bool = False) -> str:
    """Transcreve o áudio com faster-whisper (CTranslate2 + INT8 no CPU)."""
    from faster_whisper import WhisperModel
    from tqdm import tqdm

    device, compute_type = detect_device()
    mode = "rápido (beam=1)" if fast else "preciso (beam=5)"
    print(f"🧠 Carregando modelo '{model_name}' [{device.upper()} / {compute_type}] {threads} threads | modo: {mode}...")
    model = WhisperModel(
        model_name,
        device=device,
        compute_type=compute_type,
        cpu_threads=threads,
        num_workers=2,
    )

    print("🎙️  Transcrevendo...")
    segments, info = model.transcribe(
        audio_path,
        language=language,
        beam_size=1 if fast else 5,
        best_of=1 if fast else 5,
        temperature=0,
        condition_on_previous_text=False,
        without_timestamps=True,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
    )

    print(f"   Idioma detectado: {info.language} (confiança: {info.language_probability:.0%})")

    duration = round(info.duration)
    text_parts = []
    current = 0.0

    with tqdm(total=duration, unit="s", desc="Progresso", dynamic_ncols=True) as pbar:
        for seg in segments:
            text_parts.append(seg.text)
            elapsed = round(seg.end - current)
            if elapsed > 0:
                pbar.update(elapsed)
            current = seg.end

    return "".join(text_parts).strip()


OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "transcricoes")


def save_transcript(text: str, output_path: str):
    """Salva a transcrição em arquivo .txt."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"✅ Transcrição salva em: {output_path}")


def main():
    cpu_count = os.cpu_count() or 4

    parser = argparse.ArgumentParser(
        description="Transcreve vídeos do YouTube ou arquivos locais com faster-whisper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument("source", help="URL do YouTube ou caminho para arquivo de vídeo/áudio local")

    parser.add_argument(
        "--model", "-m",
        default="medium",
        choices=["tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3"],
        help="Modelo do Whisper (padrão: medium). Maior = mais preciso e mais lento."
    )

    parser.add_argument(
        "--language", "-l",
        default="pt",
        help="Idioma do vídeo (padrão: pt). Ex: en, es, fr"
    )

    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Caminho do arquivo de saída (padrão: usa o título do vídeo ou nome do arquivo)"
    )

    parser.add_argument(
        "--threads", "-t",
        type=int,
        default=cpu_count,
        help=f"Número de threads CPU (padrão: {cpu_count} — todos os núcleos lógicos)"
    )

    parser.add_argument(
        "--fast", "-f",
        action="store_true",
        help="Modo rápido: beam_size=1 (greedy). ~2x mais veloz, levemente menos preciso."
    )

    args = parser.parse_args()
    local = is_local_file(args.source)

    # 1. Checa dependências (yt-dlp só necessário para URLs)
    check_dependencies(need_ytdlp=not local)

    # 2. Define nome do arquivo de saída
    if args.output:
        # Caminho absoluto ou relativo explícito é respeitado; nome simples vai para OUTPUT_DIR
        if os.sep in args.output or "/" in args.output:
            output_path = args.output
        else:
            output_path = os.path.join(OUTPUT_DIR, args.output)
        if not output_path.endswith(".txt"):
            output_path += ".txt"
    elif local:
        base = os.path.splitext(os.path.basename(args.source))[0]
        output_path = os.path.join(OUTPUT_DIR, f"{base}.txt")
    else:
        title = get_video_title(args.source)
        output_path = os.path.join(OUTPUT_DIR, f"{title}.txt")

    # 3. Obtém o áudio e transcreve
    if local:
        print(f"📂 Usando arquivo local: {args.source}")
        text = transcribe(args.source, args.model, args.language, args.threads, args.fast)
    else:
        with tempfile.TemporaryDirectory() as tmp_dir:
            audio_path = download_audio(args.source, tmp_dir)
            text = transcribe(audio_path, args.model, args.language, args.threads, args.fast)

    # 4. Salva resultado
    save_transcript(text, output_path)

    # 5. Preview
    print("\n--- Preview ---")
    print(text[:500] + ("..." if len(text) > 500 else ""))


if __name__ == "__main__":
    main()
