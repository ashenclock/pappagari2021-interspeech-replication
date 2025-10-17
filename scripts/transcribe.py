#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

# Add project root to path to allow imports from src
sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm

from src.asr import make_backend
from src.utils.audio import AudioWalker, sanitize_model_dir, chunks


class TxtTranscriber:
    """
    Creates a single-line .txt file for each audio file.
    Output structure: <output_root>/<model_sanitized>/(ad|cn|...)/<basename>.txt
    """
    def __init__(self, audio_root: Path, output_root: Path, asr, overwrite: bool = False, verbose: bool = True):
        self.audio_root = audio_root
        self.output_root = output_root
        self.asr = asr
        self.overwrite = overwrite
        self.verbose = verbose
        self.model_dir = self.output_root / sanitize_model_dir(asr.model_name)
        self.errors = []

    def _txt_path(self, item):
        # Handle both flat and nested directory structures.
        # item.rel_under_audio_root is relative to the subdirectory (e.g., 'ad' or 'cn').
        # For flat structures, we need to handle pathing differently.
        rel_path = item.path.relative_to(self.audio_root)
        return self.model_dir / rel_path.with_suffix(".txt")

    def _ensure_parent(self, p: Path):
        p.parent.mkdir(parents=True, exist_ok=True)

    def _append_error(self, msg: str):
        self.errors.append(msg)
        if self.verbose:
            print(f"ERROR: {msg}", file=sys.stderr)

    def run(self, batch_size: int, num_workers: int):
        """Processes all audio files."""
        walker = AudioWalker(self.audio_root)
        all_items = list(walker.iter_all())

        if not all_items:
            print(f"Warning: No audio files found in {self.audio_root}", file=sys.stderr)
            return

        # For flat directories (like test set), ensure the base model dir exists
        if not any(p.is_dir() for p in self.audio_root.iterdir() if p.name not in ['.', '..']):
             self.model_dir.mkdir(parents=True, exist_ok=True)

        todo = []
        for item in all_items:
            out_path = self._txt_path(item)
            if self.overwrite or not out_path.exists():
                todo.append(item)

        if not todo:
            print("All transcripts are already up to date.")
            return

        print(f"Found {len(all_items)} audio files, transcribing {len(todo)} of them...")

        # Group items by their parent directory to preserve structure
        # This is a simplification; we process chunk by chunk.
        
        for batch_items in tqdm(list(chunks(todo, batch_size)), desc="Processing batches"):
            batch_paths = [str(item.path) for item in batch_items]
            try:
                transcripts = self.asr.transcribe_many(
                    paths=batch_paths,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    verbose=False  # We have our own progress bar
                )

                if len(transcripts) != len(batch_items):
                    self._append_error(f"Mismatch in batch results: expected {len(batch_items)}, got {len(transcripts)}")
                    continue

                for item, text in zip(batch_items, transcripts):
                    out_path = self._txt_path(item)
                    self._ensure_parent(out_path)
                    out_path.write_text(text.strip(), encoding="utf-8")

            except Exception as e:
                self._append_error(f"Failed to process batch starting with {batch_paths[0]}: {e}")

        if self.errors:
            print(f"\nCompleted with {len(self.errors)} errors.")
            error_log = self.output_root / f"errors-{sanitize_model_dir(self.asr.model_name)}.log"
            error_log.write_text("\n".join(self.errors), encoding="utf-8")
            print(f"See {error_log} for details.")
        else:
            print("\nTranscription completed successfully.")


def parse_args(config: dict | None = None):
    if config is None:
        config = {}
    
    # Extract relevant sections from config
    trans_cfg = config.get("transcription", {})
    data_cfg = config.get("data", {})

    ap = argparse.ArgumentParser(description="Generate single-line .txt transcripts using various ASR backends.")
    ap.add_argument("--dataset", default="train", choices=["train", "test"],
                    help="Select which dataset to process: 'train' or 'test' (uses paths from config).")
    ap.add_argument("--engine", default=trans_cfg.get("engine", "auto"), choices=["auto", "nemo", "whisperx", "faster-whisper"],
                    help="Select ASR backend (default: from config or 'auto').")
    ap.add_argument("--model", default=data_cfg.get("transcription_model", "large-v3"),
                    help="Model name (e.g., 'nvidia/parakeet-tdt-0.6b-v2', 'large-v2', 'large-v3').")
    ap.add_argument("--audio-root", default=data_cfg.get("audio_root"),
                    help="Root directory containing audio files (e.g., with AD/CN subfolders).")
    ap.add_argument("--output-root", default=data_cfg.get("transcripts_root", "./transcripts"),
                    help="Output directory for transcripts.")
    ap.add_argument("--device", default=trans_cfg.get("device", "cuda"), help="Device to use: 'cuda', 'cuda:0', 'cpu'.")
    ap.add_argument("--compute-type", default=trans_cfg.get("compute_type", "float16"),
                    help="Compute precision for WhisperX/faster-whisper: 'float16', 'int8', 'int8_float16'.")
    ap.add_argument("--align", action=argparse.BooleanOptionalAction, default=trans_cfg.get("align", False),
                    help="(WhisperX only) Perform word-level forced alignment.")
    ap.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=trans_cfg.get("overwrite", False),
                    help="Force regeneration of existing transcripts.")
    ap.add_argument("--batch-size", type=int, default=trans_cfg.get("batch_size", 8),
                    help="Logical batch size for processing files.")
    ap.add_argument("--num-workers", type=int, default=trans_cfg.get("num_workers", 0),
                    help="(NeMo/faster-whisper) Number of I/O workers.")
    
    args = ap.parse_args()

    # Override paths based on --dataset
    if args.dataset == "test":
        args.audio_root = data_cfg.get("test_audio_root")
        args.output_root = data_cfg.get("test_transcripts_root")
    
    return args


def main():
    # Load config from YAML
    config_path = Path(__file__).parent.parent / "config.yaml"
    config = None
    if config_path.exists():
        print(f"Loading configuration from {config_path}")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    else:
        print("Warning: config.yaml not found. Using script defaults.")

    args = parse_args(config)
    main_cli(args)


def main_cli(args):
    # Set environment variables for stability with multi-threading
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    if not args.audio_root:
        print("Error: --audio-root must be specified either in config.yaml or as a command-line argument.", file=sys.stderr)
        sys.exit(1)

    audio_root = Path(args.audio_root).expanduser().resolve()
    out_root = Path(args.output_root).expanduser().resolve()

    try:
        asr = make_backend(
            engine=args.engine,
            model_name=args.model,
            device=args.device,
            compute_type=args.compute_type,
            align=args.align,
        )
        print(f"Initialized ASR backend: {type(asr).__name__} on {asr.device} with model '{asr.model_name}'")
        
        runner = TxtTranscriber(
            audio_root=audio_root,
            output_root=out_root,
            asr=asr,
            overwrite=args.overwrite,
            verbose=True
        )
        runner.run(batch_size=max(1, args.batch_size), num_workers=max(0, args.num_workers))

    except (ImportError, ValueError) as e:
        print(f"Error initializing backend: {e}", file=sys.stderr)
        print("Please ensure the required libraries are installed for your chosen engine.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
