#!/usr/bin/env python3
from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from typing import Any, List

import torch
from tqdm import tqdm

# NeMo
try:
    import nemo.collections.asr as nemo_asr
    from nemo.collections.asr.parts.mixins.transcription import TranscribeConfig
except ImportError:
    nemo_asr = None
    TranscribeConfig = None

# WhisperX
try:
    import whisperx
except ImportError:
    whisperx = None

# faster-whisper
try:
    from faster_whisper import WhisperModel as FasterWhisperModel
except ImportError:
    FasterWhisperModel = None


class BaseASR(ABC):
    """Abstract Base Class for ASR models."""
    model_name: str
    device: str

    @abstractmethod
    def transcribe_many(self, paths: List[str], batch_size: int, num_workers: int, verbose: bool) -> List[str]:
        """Transcribe a list of audio files."""
        raise NotImplementedError

    @staticmethod
    def _extract_text(item: Any) -> str:
        """Extracts the transcribed text from a result object."""
        if isinstance(item, str):
            return item
        if hasattr(item, "text"):
            return item.text
        # Add more specific extraction logic if needed for other backends
        return str(item)


class NeMoASR(BaseASR):
    """ASR backend using NVIDIA NeMo."""
    def __init__(self, model_name: str, device: str = "cuda"):
        if nemo_asr is None or TranscribeConfig is None:
            raise ImportError("NeMo is not installed. Please install it with 'pip install nemo_toolkit[asr]'")
        self.model_name = model_name
        self.requested_device = device

        self.impl = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)
        target = "cpu"
        if device.startswith("cuda") and torch.cuda.is_available():
            target = device
        self.device = target
        self.impl = self.impl.to(torch.device(target)).eval()

    def _call_transcribe(self, paths: List[str], batch_size: int, num_workers: int, verbose: bool) -> List[str]:
        cfg = TranscribeConfig(
            batch_size=batch_size,
            num_workers=num_workers,
            channel_selector="average",  # You might want to configure this
            verbose=verbose,
        )
        out = self.impl.transcribe(paths, override_config=cfg)
        if isinstance(out, tuple) and len(out) == 2:
            out = out[0]  # Often returns (transcriptions, WER)
        return [self._extract_text(x).strip() for x in out]

    def transcribe_many(self, paths: List[str], batch_size: int, num_workers: int, verbose: bool) -> List[str]:
        try:
            return self._call_transcribe(paths, batch_size, num_workers, verbose)
        except Exception:
            # Fallback for models that don't support batching well
            print(f"WARN: Batch transcription failed for {self.model_name}. Falling back to one-by-one.", file=sys.stderr)
            results = []
            for p in paths:
                results.extend(self._call_transcribe([p], 1, 0, verbose))
            return results


class WhisperXASR(BaseASR):
    """ASR backend using WhisperX."""
    def __init__(self, model_name: str, device: str = "cuda", compute_type: str = "float16", align: bool = False):
        if whisperx is None:
            raise ImportError("WhisperX is not installed. Please install it with 'pip install git+https://github.com/m-bain/whisperx.git'")
        self.model_name = model_name
        self.requested_device = device
        self.align = align
        
        target_device = "cpu"
        if device.startswith("cuda") and torch.cuda.is_available():
            target_device = device
        self.device = target_device

        self.model = whisperx.load_model(model_name, device=self.device, compute_type=compute_type)
        self.aligner = None
        self.metadata = None

    def _ensure_aligner(self, language_code: str):
        if self.aligner is None or self.metadata is None:
            self.aligner, self.metadata = whisperx.load_align_model(language_code=language_code, device=self.device)

    @staticmethod
    def _join_segments(segments: List[dict]) -> str:
        return " ".join(s["text"].strip() for s in segments).strip()

    def _transcribe_one(self, path: str, batch_size: int) -> str:
        audio = whisperx.load_audio(path)
        result = self.model.transcribe(audio, batch_size=batch_size)
        
        if self.align:
            self._ensure_aligner(result["language"])
            result = whisperx.align(result["segments"], self.aligner, self.metadata, audio, self.device)
        
        return self._join_segments(result["segments"])

    def transcribe_many(self, paths: List[str], batch_size: int, num_workers: int, verbose: bool) -> List[str]:
        # WhisperX's batching is for segments of one audio, not multiple files.
        # We process files one by one, but can use the internal batch_size.
        results = []
        for path in (tqdm(paths, desc="Transcribing") if verbose else paths):
            results.append(self._transcribe_one(path, batch_size))
        return results


class FasterWhisperASR(BaseASR):
    """ASR backend using faster-whisper."""
    def __init__(self, model_name: str, device: str = "cuda", compute_type: str = "float16"):
        if FasterWhisperModel is None:
            raise ImportError("faster-whisper is not installed. Please install it with 'pip install faster-whisper'")
        self.model_name = model_name
        self.requested_device = device
        
        target_device = "cpu"
        if device.startswith("cuda") and torch.cuda.is_available():
            target_device = "cuda" # faster-whisper doesn't take device index
        self.device = target_device

        self.model = FasterWhisperModel(model_name, device=self.device, compute_type=compute_type)

    def transcribe_many(self, paths: List[str], batch_size: int, num_workers: int, verbose: bool) -> List[str]:
        # faster-whisper has its own batching mechanism via `transcribe_files`
        segments_generator, _ = self.model.transcribe_files(
            paths,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        
        results = []
        current_path_index = 0
        current_text = []

        # This logic assumes the generator yields segments for each file in order.
        # It's a bit complex because we need to group segments by file.
        # A simpler, but potentially slower way, would be to loop and call `transcribe` per file.
        # For now, let's stick to the more efficient `transcribe_files`.
        
        # Let's simplify and process one by one to guarantee correctness.
        # The performance gain from transcribing multiple files at once is often complex to manage.
        all_transcripts = []
        for path in (tqdm(paths, desc="Transcribing") if verbose else paths):
            segments, _ = self.model.transcribe(path)
            full_text = " ".join(s.text.strip() for s in segments).strip()
            all_transcripts.append(full_text)
        return all_transcripts


def make_backend(engine: str, model_name: str, device: str, compute_type: str, align: bool) -> BaseASR:
    """Factory function to create an ASR backend."""
    eng = engine.lower()
    if eng == "auto":
        if "nvidia/" in model_name:
            eng = "nemo"
        else:
            eng = "whisperx"  # Default to WhisperX if not a NeMo model
    
    if eng == "nemo":
        return NeMoASR(model_name, device)
    if eng == "whisperx":
        return WhisperXASR(model_name, device, compute_type, align)
    if eng == "faster-whisper":
        return FasterWhisperASR(model_name, device, compute_type)
    
    raise ValueError(f"Invalid engine: {engine}")
