#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple
import os

import soundfile as sf
from tqdm import tqdm

# --- ASR backend (Whisper) ----------------------------------------------------
try:
    import whisper
except Exception as e:
    raise SystemExit("Whisper non installato. Esegui: pip install openai-whisper") from e


class WhisperBackend:
    def __init__(self, model_name: str = "large-v3", device: Optional[str] = "cuda"):
        self.model_name = model_name
        self.device = device
        self.model = whisper.load_model(model_name, device=device)

    def transcribe_file(self, path: Path) -> str:
        res = self.model.transcribe(str(path))
        return (res or {}).get("text", "").strip()


# --- dataset walker ------------------------------------------------------------
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma"}

@dataclass
class AudioItem:
    path: Path
    rel_under_audio_root: Path   # es. ad/adrso024.wav


class AudioWalker:
    def __init__(self, audio_root: Path):
        self.audio_root = audio_root

    def _iter_dir(self, cls_dir: Path) -> Iterable[AudioItem]:
        if not cls_dir.exists():
            return
        for p in sorted(cls_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
                yield AudioItem(path=p, rel_under_audio_root=p.relative_to(self.audio_root))

    def iter_all(self) -> Iterable[AudioItem]:
        for sub in ("ad", "cn"):
            yield from self._iter_dir(self.audio_root / sub)


# --- transcriber runner ---------------------------------------------------------
class TxtTranscriber:
    """
    Crea .txt con la trascrizione completa (una riga di testo) per ogni audio.
    Output root = <output_root>/<model_name_sanitized>/(ad|cn)/<basename>.txt
    """
    def __init__(
        self,
        audio_root: Path,
        output_root: Path,
        asr: WhisperBackend,
        overwrite: bool = False,
        model_name_for_dir: str = "openai-whisper-large-v3",
    ):
        self.audio_root = audio_root
        self.output_root = output_root
        self.asr = asr
        self.overwrite = overwrite
        # sanitize nome dir (evita slash)
        self.model_dir = model_name_for_dir.replace("/", "-")
        self.target_root = self.output_root / self.model_dir

    def _txt_path_for(self, item: AudioItem) -> Path:
        rel = item.rel_under_audio_root.with_suffix(".txt")  # ad/xyz.txt
        return self.target_root / rel

    def _ensure_parent(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

    def run(self) -> None:
        walker = AudioWalker(self.audio_root)
        items = list(walker.iter_all())
        pbar = tqdm(items, desc=f"Transcribing → {self.model_dir}", unit="file")
        for item in pbar:
            out_txt = self._txt_path_for(item)
            if out_txt.exists() and not self.overwrite:
                pbar.set_postfix(skipped=True); continue

            # quick sanity (durata)
            try:
                info = sf.info(str(item.path))
                _ = float(info.frames) / (info.samplerate or 1)
            except Exception:
                pass

            text = self.asr.transcribe_file(item.path)
            self._ensure_parent(out_txt)
            out_txt.write_text(text + "\n", encoding="utf-8")
            pbar.set_postfix(saved=str(out_txt.relative_to(self.output_root)))


# --- CLI -----------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Trascrizione completa in .txt (Whisper large-v3).")
    ap.add_argument("--audio-root", required=True,
                    help="Radice con le cartelle audio/ad e audio/cn (es. .../diagnosis/train/audio)")
    ap.add_argument("--output-root", default="./transcripts",
                    help="Cartella in cui creare <model_name>/ad|cn/*.txt (default: ./transcripts)")
    ap.add_argument("--device", default="cuda", help="cuda | cpu")
    ap.add_argument("--overwrite", action="store_true", help="Rigenera i .txt se già presenti.")
    # nome per la cartella (stile HF): user: 'usa whisperv3' → usiamo 'openai-whisper-large-v3'
    ap.add_argument("--model-dir-name", default="openai-whisper-large-v3",
                    help="Nome cartella del modello (stile HF), default=openai-whisper-large-v3")
    args = ap.parse_args()

    audio_root = Path(args.audio_root).expanduser().resolve()
    out_root = Path(args.output_root).expanduser().resolve()

    asr = WhisperBackend(model_name="large-v3", device=args.device)
    runner = TxtTranscriber(
        audio_root=audio_root,
        output_root=out_root,
        asr=asr,
        overwrite=args.overwrite,
        model_name_for_dir=args.model_dir_name,
    )
    runner.run()
