#!/usr/bin/env python3
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma"}

def sanitize_model_dir(name: str) -> str:
    """Remove special characters from a model name to create a valid directory name."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", name)

def chunks(seq, n):
    """Yield successive n-sized chunks from a sequence."""
    for i in range(0, len(seq), n):
        yield seq[i : i + n]

@dataclass
class AudioItem:
    """Represents an audio file with its path and relative path."""
    path: Path
    rel_under_audio_root: Path

class AudioWalker:
    """Walks an audio directory structure to find audio files."""
    def __init__(self, audio_root: Path):
        self.audio_root = audio_root

    def _iter_tree(self, root: Path) -> Iterable[AudioItem]:
        """Recursively find all audio files under a root directory."""
        for p in sorted(root.rglob("*")):
            if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
                rel_path = p.relative_to(self.audio_root)
                yield AudioItem(path=p, rel_under_audio_root=rel_path)

    def iter_all(self) -> Iterable[AudioItem]:
        """
        Iterate through all audio files, handling 'ad' and 'cn' subdirectories
        or a flat structure.
        """
        subs = {d.name.lower(): d for d in self.audio_root.iterdir() if d.is_dir()} if self.audio_root.exists() else {}
        used = False
        for key in ("ad", "cn"):
            if key in subs:
                used = True
                yield from self._iter_tree(subs[key])
        if not used:
            yield from self._iter_tree(self.audio_root)
