#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from typing import Iterable, Optional
import os, math
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from tqdm import tqdm
from dotenv import load_dotenv

# ------------------ HF ASR backend (unificato) ------------------

class UnifiedHFASR:
    """
    Backend unico basato su 🤗 Transformers pipeline('automatic-speech-recognition').
    Accetta qualsiasi modello ASR su HF (es. 'openai/whisper-large-v3', 'nvidia/parakeet-tdt-0.6b-v2').
    Carica HF_TOKEN da .env se presente.
    Passa alla pipeline un dict {'array': wav, 'sampling_rate': sr} => niente ffmpeg necessario.
    """
    def __init__(self, model_name: str, device: Optional[str] = "cuda"):
        from transformers import pipeline
        load_dotenv()
        token = os.getenv("HF_TOKEN")  # opzionale per modelli pubblici; necessario se gated
        self.model_name = model_name

        # mappa device stringa -> indice numerico per Transformers
        device_id = -1
        if device:
            if device == "cuda":
                device_id = 0
            elif device.startswith("cuda:"):
                device_id = int(device.split(":")[1])
            elif device == "cpu":
                device_id = -1

        self.pipe = pipeline(
            task="automatic-speech-recognition",
            model=model_name,
            use_auth_token=token,
            device=device_id,
        )

    @staticmethod
    def _load_mono(path: Path) -> tuple[np.ndarray, int]:
        wav, sr = sf.read(str(path), always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        wav = wav.astype(np.float32)
        return wav, sr

    @staticmethod
    def _resample(wav: np.ndarray, sr: int, target_sr: int) -> tuple[np.ndarray, int]:
        if sr == target_sr or sr <= 0:
            return wav, sr
        g = math.gcd(sr, target_sr)
        wav = resample_poly(wav, target_sr // g, sr // g).astype(np.float32)
        return wav, target_sr

    def transcribe_file(self, path: Path, target_sr: int = 16000) -> str:
        wav, sr = self._load_mono(path)
        wav, sr = self._resample(wav, sr, target_sr)
        # Passiamo array + sampling_rate, evitando dipendenze su ffmpeg
        out = self.pipe({"array": wav, "sampling_rate": sr})
        if isinstance(out, dict):
            return (out.get("text") or out.get("generated_text") or "").strip()
        return str(out).strip()

# ------------------ dataset walker ------------------

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma"}

@dataclass
class AudioItem:
    path: Path
    rel_under_audio_root: Path   # es. ad/adrso024.wav

class AudioWalker:
    def __init__(self, audio_root: Path):
        self.audio_root = audio_root

    def _iter_dir(self, cls_dir: Path):
        if not cls_dir.exists(): return
        for p in sorted(cls_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
                yield AudioItem(path=p, rel_under_audio_root=p.relative_to(self.audio_root))

    def iter_all(self):
        for sub in ("ad", "cn"):
            yield from self._iter_dir(self.audio_root / sub)

# ------------------ runner ------------------

class TxtTranscriber:
    """
    Crea .txt con la trascrizione completa per ogni audio.
    Output: <output_root>/<sanitized_model_name>/(ad|cn)/<basename>.txt
    """
    def __init__(self, audio_root: Path, output_root: Path, asr_backend: UnifiedHFASR, overwrite: bool = False):
        self.audio_root = audio_root
        self.output_root = output_root
        self.asr = asr_backend
        self.overwrite = overwrite
        self.model_dir = self._sanitize(self.asr.model_name)
        self.target_root = self.output_root / self.model_dir

    @staticmethod
    def _sanitize(model_name: str) -> str:
        return model_name.replace("/", "-")

    def _txt_path_for(self, item: AudioItem) -> Path:
        rel = item.rel_under_audio_root.with_suffix(".txt")  # ad/xyz.txt
        return self.target_root / rel

    def _ensure_parent(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

    def run(self) -> None:
        items = list(AudioWalker(self.audio_root).iter_all())
        desc = f"Transcribing → {self.model_dir}"
        for item in tqdm(items, desc=desc, unit="file"):
            out_txt = self._txt_path_for(item)
            if out_txt.exists() and not self.overwrite:
                continue
            text = self.asr.transcribe_file(item.path)
            self._ensure_parent(out_txt)
            out_txt.write_text(text + "\n", encoding="utf-8")

# ------------------ CLI ------------------

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Trascrizione completa .txt con Transformers ASR (modello HF a scelta).")
    ap.add_argument("--audio-root", required=True,
                    help="Radice con le cartelle audio/ad e audio/cn (es. .../diagnosis/train/audio)")
    ap.add_argument("--output-root", default="./transcripts",
                    help="Cartella out (default: ./transcripts)")
    ap.add_argument("--model", default="openai/whisper-large-v3",
                    help="Modello HF (es: openai/whisper-large-v3, nvidia/parakeet-tdt-0.6b-v2)")
    ap.add_argument("--device", default="cuda", help="cuda | cuda:0 | cpu")
    ap.add_argument("--overwrite", action="store_true", help="Rigenera i .txt se già presenti.")
    ap.add_argument("--target-sr", type=int, default=16000, help="Sample rate da usare per l’ASR")
    args = ap.parse_args()

    audio_root = Path(args.audio_root).expanduser().resolve()
    out_root = Path(args.output_root).expanduser().resolve()

    asr = UnifiedHFASR(model_name=args.model, device=args.device)
    runner = TxtTranscriber(audio_root=audio_root, output_root=out_root, asr_backend=asr, overwrite=args.overwrite)
    # opzionale: se vuoi passare target_sr dinamico, aggiungi un arg e inoltralo a transcribe_file
    runner.run()
