#!/usr/bin/env python3
"""
transcribe.py — Transcreve vídeos do YouTube ou arquivos locais usando yt-dlp + faster-whisper

Uso:
    python transcribe.py <URL ou arquivo> [opções]
    python transcribe.py <URL1> <URL2> <URL3> ...          # lote via CLI
    python transcribe.py --batch lista.txt                  # lote via arquivo

Exemplos:
    python transcribe.py https://youtu.be/xxxxxxxxxxx
    python transcribe.py video.mp4
    python transcribe.py https://youtu.be/xxxxxxxxxxx --model large-v2
    python transcribe.py video.mp4 --output minha_transcricao.txt
    python transcribe.py https://youtu.be/xxxxxxxxxxx --language en
    python transcribe.py URL1 URL2 URL3 --model medium --fast
    python transcribe.py --batch urls.txt --language en
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
        "--audio-format", "m4a",   # m4a: formato nativo do YouTube, remux sem re-encoding
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


def _sanitize(name: str, max_len: int = 80) -> str:
    """Remove caracteres inválidos para nomes de arquivo/pasta no Windows."""
    invalid = r'\/:*?"<>|'
    for char in invalid:
        name = name.replace(char, "_")
    return name[:max_len].strip()


def get_video_info(url: str) -> tuple[str, str]:
    """Retorna (canal, título) do vídeo. Fallbacks seguros em caso de erro."""
    cmd = [
        get_yt_dlp_cmd(),
        "--no-playlist",
        "--print", "%(channel)s\t%(title)s",
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        parts = result.stdout.strip().split("\t", 1)
        if len(parts) == 2:
            channel = _sanitize(parts[0]) or "desconhecido"
            title = _sanitize(parts[1]) or "transcricao"
            return channel, title

    return "desconhecido", "transcricao"


def load_model(model_name: str, threads: int, fast: bool):
    """Carrega o modelo Whisper uma vez para reutilizar em múltiplas transcrições.

    Retorna (model, batch_size) onde batch_size é None no modo padrão ou um
    inteiro quando BatchedInferencePipeline está disponível.
    """
    from faster_whisper import WhisperModel

    device, compute_type = detect_device()
    num_workers = min(4, os.cpu_count() or 2)
    mode = "rapido (beam=1)" if fast else "preciso (beam=5)"
    print(f"🧠 Carregando modelo '{model_name}' [{device.upper()} / {compute_type}] {threads} threads | modo: {mode}...")

    base_model = WhisperModel(
        model_name,
        device=device,
        compute_type=compute_type,
        cpu_threads=threads,
        num_workers=num_workers,
    )

    if not fast and device == "cuda":
        try:
            from faster_whisper import BatchedInferencePipeline
            print("   Inferencia em lote ativada (batch_size=16, CUDA)")
            return BatchedInferencePipeline(model=base_model), 16
        except ImportError:
            pass

    return base_model, None


def transcribe(audio_path: str, model, language: str, fast: bool = False, batch_size: int | None = None) -> str:
    """Transcreve o áudio com um modelo faster-whisper já carregado."""
    from tqdm import tqdm

    print("🎙️  Transcrevendo...")

    if batch_size is not None:
        segments, info = model.transcribe(
            audio_path,
            language=language,
            beam_size=1 if fast else 5,
            batch_size=batch_size,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )
    else:
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


def resolve_output_path(source: str, explicit_output: str | None, batch_mode: bool) -> str:
    """Determina o caminho do arquivo de saída para uma fonte."""
    if explicit_output and not batch_mode:
        if os.sep in explicit_output or "/" in explicit_output:
            path = explicit_output
        else:
            path = os.path.join(OUTPUT_DIR, explicit_output)
        return path if path.endswith(".txt") else path + ".txt"

    if is_local_file(source):
        base = os.path.splitext(os.path.basename(source))[0]
        return os.path.join(OUTPUT_DIR, "local", f"{base}.txt")

    channel, title = get_video_info(source)
    return os.path.join(OUTPUT_DIR, channel, f"{title}.txt")


def process_source(source: str, output_path: str, model, batch_size: int | None, language: str, fast: bool):
    """Baixa (se URL) e transcreve uma única fonte. Retorna True em caso de sucesso."""
    local = is_local_file(source)
    try:
        if local:
            print(f"\n📂 Arquivo local: {source}")
            text = transcribe(source, model, language, fast, batch_size)
        else:
            channel = os.path.basename(os.path.dirname(output_path))
            print(f"\n📺 Canal: {channel}")
            with tempfile.TemporaryDirectory() as tmp_dir:
                audio_path = download_audio(source, tmp_dir)
                text = transcribe(audio_path, model, language, fast, batch_size)

        save_transcript(text, output_path)
        print("\n--- Preview ---")
        print(text[:500] + ("..." if len(text) > 500 else ""))
        return True
    except SystemExit:
        return False
    except Exception as exc:
        print(f"❌ Erro ao processar '{source}': {exc}")
        return False


def _remove_from_batch(batch_file: str | None, source: str):
    """Remove a linha correspondente à fonte do arquivo de lote."""
    if not batch_file or not os.path.isfile(batch_file):
        return
    with open(batch_file, encoding="utf-8") as f:
        lines = f.readlines()
    with open(batch_file, "w", encoding="utf-8") as f:
        for line in lines:
            if line.strip() != source:
                f.write(line)


def main():
    cpu_count = os.cpu_count() or 4

    parser = argparse.ArgumentParser(
        description="Transcreve vídeos do YouTube ou arquivos locais com faster-whisper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "sources",
        nargs="*",
        help="URL(s) do YouTube ou caminho(s) para arquivo(s) local(is). Aceita múltiplos."
    )

    parser.add_argument(
        "--batch", "-b",
        metavar="ARQUIVO",
        help="Arquivo de texto com uma URL ou caminho por linha (linhas com # são ignoradas)."
    )

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
        help="Caminho do arquivo de saída (só válido para fonte única; ignorado em lote)."
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

    # Consolida lista de fontes
    all_sources = list(args.sources)
    batch_file = args.batch
    if batch_file:
        if not os.path.isfile(batch_file):
            print(f"❌ Arquivo de lote não encontrado: {batch_file}")
            sys.exit(1)
        with open(batch_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    all_sources.append(line)

    if not all_sources:
        parser.print_help()
        sys.exit(1)

    batch_mode = len(all_sources) > 1

    if args.output and batch_mode:
        print("⚠️  --output ignorado no modo lote; os nomes são gerados automaticamente.\n")

    # Verifica dependências (yt-dlp se houver alguma URL)
    has_urls = any(not is_local_file(s) for s in all_sources)
    check_dependencies(need_ytdlp=has_urls)

    # Carrega o modelo uma única vez para todos os vídeos
    model, batch_size = load_model(args.model, args.threads, args.fast)

    ok, failed = 0, []

    for i, source in enumerate(all_sources, 1):
        if batch_mode:
            print(f"\n{'='*60}")
            print(f"[{i}/{len(all_sources)}] {source}")
            print('='*60)

        output_path = resolve_output_path(source, args.output, batch_mode)

        if os.path.isfile(output_path):
            print(f"⏭️  Já transcrito, ignorando: {output_path}")
            _remove_from_batch(batch_file, source)
            ok += 1
            continue

        success = process_source(source, output_path, model, batch_size, args.language, args.fast)
        if success:
            ok += 1
            _remove_from_batch(batch_file, source)
        else:
            failed.append(source)

    if batch_mode:
        print(f"\n{'='*60}")
        print(f"Lote concluído: {ok}/{len(all_sources)} transcrições salvas.")
        if failed:
            print("Falhas:")
            for s in failed:
                print(f"  - {s}")
        print('='*60)


if __name__ == "__main__":
    main()
