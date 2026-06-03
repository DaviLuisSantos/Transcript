#!/usr/bin/env python3
"""
transcribe.py — Transcreve vídeos do YouTube usando yt-dlp + faster-whisper

Uso:
    python transcribe.py <URL> [opções]

Exemplos:
    python transcribe.py https://youtu.be/xxxxxxxxxxx
    python transcribe.py https://youtu.be/xxxxxxxxxxx --model large-v2
    python transcribe.py https://youtu.be/xxxxxxxxxxx --output minha_transcricao.txt
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


def check_dependencies():
    """Verifica se faster-whisper e yt-dlp estão instalados."""
    missing = []

    try:
        import faster_whisper
    except ImportError:
        missing.append("faster-whisper")

    try:
        result = subprocess.run([get_yt_dlp_cmd(), "--version"], capture_output=True)
        if result.returncode != 0:
            missing.append("yt-dlp")
    except FileNotFoundError:
        missing.append("yt-dlp")

    if missing:
        print("❌ Dependências faltando. Instale com:")
        print(f"   pip install {' '.join(missing)}")
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

    device, compute_type = detect_device()
    mode = "rápido (beam=1)" if fast else "preciso (beam=5)"
    print(f"🧠 Carregando modelo '{model_name}' [{device.upper()} / {compute_type}] {threads} threads | modo: {mode}...")
    model = WhisperModel(
        model_name,
        device=device,
        compute_type=compute_type,
        cpu_threads=threads,
        num_workers=2,          # paraleliza carregamento de segmentos
    )

    print("🎙️  Transcrevendo...")
    segments, info = model.transcribe(
        audio_path,
        language=language,
        beam_size=1 if fast else 5,          # beam_size=1 = greedy, ~2x mais rápido
        best_of=1 if fast else 5,
        temperature=0,                        # desativa fallback de temperatura (mais rápido)
        condition_on_previous_text=False,     # evita re-processar contexto anterior
        vad_filter=True,                      # pula silêncio automaticamente
        vad_parameters=dict(min_silence_duration_ms=500),
    )

    print(f"   Idioma detectado: {info.language} (confiança: {info.language_probability:.0%})")

    # Consome o gerador mostrando progresso
    import time
    duration = info.duration
    text_parts = []
    last_print = 0.0

    for seg in segments:
        text_parts.append(seg.text)
        pct = seg.end / duration * 100 if duration else 0
        if pct - last_print >= 10:
            print(f"   {pct:5.1f}%  [{seg.start:6.1f}s → {seg.end:6.1f}s]  {seg.text[:60].strip()}...")
            last_print = pct

    return "".join(text_parts).strip()


def save_transcript(text: str, output_path: str):
    """Salva a transcrição em arquivo .txt."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"✅ Transcrição salva em: {output_path}")


def main():
    import os
    cpu_count = os.cpu_count() or 4

    parser = argparse.ArgumentParser(
        description="Transcreve vídeos do YouTube com faster-whisper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument("url", help="URL do vídeo do YouTube")

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
        help="Caminho do arquivo de saída (padrão: usa o título do vídeo)"
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

    # 1. Checa dependências
    check_dependencies()

    # 2. Define nome do arquivo de saída
    if args.output:
        output_path = args.output
        if not output_path.endswith(".txt"):
            output_path += ".txt"
    else:
        title = get_video_title(args.url)
        output_path = f"{title}.txt"

    # 3. Baixa áudio em diretório temporário
    with tempfile.TemporaryDirectory() as tmp_dir:
        audio_path = download_audio(args.url, tmp_dir)

        # 4. Transcreve
        text = transcribe(audio_path, args.model, args.language, args.threads, args.fast)

    # 5. Salva resultado
    save_transcript(text, output_path)

    # 6. Preview
    print("\n--- Preview ---")
    print(text[:500] + ("..." if len(text) > 500 else ""))


if __name__ == "__main__":
    main()
