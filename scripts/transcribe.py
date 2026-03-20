"""ASR transcription using WhisperX, NeMo, or CrisperWhisper."""

import argparse
import sys
from pathlib import Path
from tqdm import tqdm
import torch
import gc

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import load_config


def _get_audio_files(audio_dir, output_dir, overwrite):
    """Get list of audio files to transcribe, skipping already-transcribed."""
    audio_files = sorted(Path(audio_dir).rglob("*.wav"))
    if not overwrite:
        audio_files = [f for f in audio_files if not (output_dir / f"{f.stem}.txt").exists()]
    return audio_files


def _save_transcript(output_dir, stem, text):
    out_path = output_dir / f"{stem}.txt"
    out_path.write_text(text.strip(), encoding='utf-8')


def get_crisperwhisper_transcriber(config):
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    cfg = config.transcription.crisperwhisper
    device = config.device
    torch_dtype = torch.float16 if "cuda" in device else torch.float32

    print(f"Loading CrisperWhisper '{cfg.model_id}'...")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(cfg.model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True)
    model.to(device)
    processor = AutoProcessor.from_pretrained(cfg.model_id)
    pipe = pipeline(
        "automatic-speech-recognition", model=model, tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor, chunk_length_s=30,
        batch_size=cfg.batch_size, torch_dtype=torch_dtype, device=device,
    )

    def transcribe_files(audio_dir, output_dir):
        audio_files = _get_audio_files(audio_dir, output_dir, config.transcription.overwrite)
        if not audio_files:
            print(f"Nothing to transcribe in {audio_dir}")
            return
        print(f"Transcribing {len(audio_files)} files from {audio_dir}...")
        for audio_path in tqdm(audio_files, desc=f"Transcribing {Path(audio_dir).name}"):
            try:
                result = pipe(str(audio_path))
                _save_transcript(output_dir, audio_path.stem, result["text"])
            except Exception as e:
                print(f"\nERROR on {audio_path.name}: {e}")

    gc.collect()
    torch.cuda.empty_cache()
    return transcribe_files


def get_nemo_transcriber(config):
    try:
        import nemo.collections.asr as nemo_asr
    except ImportError:
        print("ERROR: NVIDIA NeMo Toolkit not installed.")
        sys.exit(1)

    cfg = config.transcription.nemo
    print(f"Loading NeMo '{cfg.model_name}'...")
    backend = nemo_asr.models.ASRModel.from_pretrained(model_name=cfg.model_name)
    backend.to(torch.device(config.device))

    def transcribe_files(audio_dir, output_dir):
        audio_files = _get_audio_files(audio_dir, output_dir, config.transcription.overwrite)
        if not audio_files:
            print(f"Nothing to transcribe in {audio_dir}")
            return
        print(f"Transcribing {len(audio_files)} files from {audio_dir}...")
        result = backend.transcribe(audio=[str(p) for p in audio_files], batch_size=cfg.batch_size, channel_selector=0, verbose=True)
        transcriptions = result[0] if isinstance(result, tuple) else result
        for audio_path, text_obj in zip(audio_files, transcriptions):
            final_text = text_obj.text if hasattr(text_obj, 'text') else text_obj
            _save_transcript(output_dir, audio_path.stem, final_text or "")

    return transcribe_files


def get_whisperx_transcriber(config):
    try:
        import whisperx
    except ImportError:
        print("ERROR: 'whisperx' not installed.")
        sys.exit(1)

    cfg = config.transcription.whisperx
    print(f"Loading WhisperX '{cfg.model_name}'...")
    model = whisperx.load_model(cfg.model_name, config.device, compute_type=cfg.compute_type, language=cfg.language)

    def transcribe_files(audio_dir, output_dir):
        audio_files = _get_audio_files(audio_dir, output_dir, config.transcription.overwrite)
        if not audio_files:
            print(f"Nothing to transcribe in {audio_dir}")
            return
        print(f"Transcribing {len(audio_files)} files from {audio_dir}...")
        for audio_path in tqdm(audio_files, desc=f"Transcribing {Path(audio_dir).name}"):
            try:
                audio = whisperx.load_audio(str(audio_path))
                result = model.transcribe(audio, batch_size=cfg.batch_size)
                full_text = " ".join(seg['text'].strip() for seg in result["segments"])
                _save_transcript(output_dir, audio_path.stem, full_text)
            except Exception as e:
                print(f"\nERROR on {audio_path.name}: {e}")

    gc.collect()
    torch.cuda.empty_cache()
    return transcribe_files


_ENGINE_MAP = {
    "crisperwhisper": get_crisperwhisper_transcriber,
    "nemo": get_nemo_transcriber,
    "whisperx": get_whisperx_transcriber,
}

_ENGINE_MODEL_NAME = {
    "crisperwhisper": lambda c: c.transcription.crisperwhisper.model_id.split('/')[-1],
    "nemo": lambda c: c.transcription.nemo.model_name.split('/')[-1],
    "whisperx": lambda c: f"WhisperX_{c.transcription.whisperx.model_name}",
}


def main():
    parser = argparse.ArgumentParser(description="Generate transcriptions.")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    engine = config.transcription.engine.lower()

    if engine not in _ENGINE_MAP:
        print(f"ERROR: Engine '{engine}' not supported. Choose from: {list(_ENGINE_MAP.keys())}")
        sys.exit(1)

    transcribe_fn = _ENGINE_MAP[engine](config)
    model_folder = _ENGINE_MODEL_NAME[engine](config)

    train_output_dir = Path(config.data.transcripts_root) / model_folder
    test_output_dir = Path(config.data.test_transcripts_root) / model_folder
    train_output_dir.mkdir(parents=True, exist_ok=True)
    test_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n--- Transcribing Training Set ({engine}) ---")
    transcribe_fn(Path(config.data.audio_root), train_output_dir)

    print(f"\n--- Transcribing Test Set ({engine}) ---")
    transcribe_fn(Path(config.data.test_audio_root), test_output_dir)

    print("\nTranscription complete.")


if __name__ == "__main__":
    main()
