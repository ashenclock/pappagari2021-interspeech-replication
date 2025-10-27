import argparse
import sys
from pathlib import Path
from tqdm import tqdm
import torch
import gc

# Aggiunge la root del progetto al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config

# ===================================================================
#      IMPLEMENTAZIONE CORRETTA PER NYRAHEALTH/CRISPERWHISPER
# ===================================================================
def get_crisperwhisper_transcriber(config):
    """
    Prepara la pipeline per CrisperWhisper.
    Carica il token dal file .env solo per questo engine.
    """
    # --- MODIFICA CHIAVE: IMPORT E CARICAMENTO LOCALE ---
    # Questo codice viene eseguito SOLO se l'engine è "crisperwhisper".
    # In questo modo, gli altri ambienti non hanno bisogno di 'python-dotenv'.
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("Info: File .env trovato e caricato per l'autenticazione a Hugging Face.")
    except ImportError:
        print("Info: Libreria 'python-dotenv' non trovata. Si procederà usando il token salvato da 'huggingface-cli login' se disponibile.")
    # --- FINE MODIFICA ---

    try:
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
    except ImportError:
        print("\nERRORE: La libreria 'transformers' (versione custom) non è installata.")
        print("Assicurati di aver attivato 'crisper_env' e di aver eseguito 'pip install git+...'")
        sys.exit(1)

    cfg = config.transcription.crisperwhisper
    device = config.device
    torch_dtype = torch.float16 if "cuda" in device else torch.float32
    
    print(f"Caricamento backend CrisperWhisper (HF Pipeline) con modello '{cfg.model_id}'...")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        cfg.model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(cfg.model_id)
    transcription_pipeline = pipeline(
        "automatic-speech-recognition", model=model, tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor, chunk_length_s=30,
        batch_size=cfg.batch_size, torch_dtype=torch_dtype, device=device,
    )
    print("Backend CrisperWhisper inizializzato.")

    def transcribe_files(audio_dir, output_dir):
        # ... (Questa logica interna rimane invariata)
        audio_files = sorted([p for p in Path(audio_dir).rglob("*.wav")])
        if not config.transcription.overwrite:
            audio_files = [f for f in audio_files if not (output_dir / f"{f.stem}.txt").exists()]
        if not audio_files:
            print(f"Nessun file da trascrivere in {audio_dir} (o già trascritti).")
            return
        print(f"Trascrizione di {len(audio_files)} file da {audio_dir}...")
        for audio_path in tqdm(audio_files, desc=f"Transcribing {audio_dir.name}"):
            try:
                result = transcription_pipeline(str(audio_path))
                full_text = result["text"].strip()
                out_path = output_dir / f"{audio_path.stem}.txt"
                out_path.write_text(full_text, encoding='utf-8')
            except Exception as e:
                print(f"\nERRORE durante la trascrizione di {audio_path.name}: {e}")
                continue
    
    gc.collect()
    torch.cuda.empty_cache()
    return transcribe_files

# ===================================================================
#             LOGICA PER NEMO E WHISPERX (INVARIATE)
# ===================================================================

def get_nemo_transcriber(config):
    # ... (codice identico a prima, nessuna modifica)
    try:
        import nemo.collections.asr as nemo_asr
    except ImportError:
        print("\nERRORE: NVIDIA NeMo Toolkit non è installato. Attiva l'ambiente 'nemo_env'.")
        sys.exit(1)
    cfg = config.transcription.nemo
    print(f"Caricamento backend NeMo con modello '{cfg.model_name}'...")
    backend = nemo_asr.models.ASRModel.from_pretrained(model_name=cfg.model_name)
    backend.to(torch.device(config.device))
    def transcribe_files(audio_dir, output_dir):
        audio_files = sorted([p for p in Path(audio_dir).rglob("*.wav")])
        if not config.transcription.overwrite:
            audio_files = [f for f in audio_files if not (output_dir / f"{f.stem}.txt").exists()]
        if not audio_files:
            print(f"Nessun file da trascrivere in {audio_dir} (o sono già tutti presenti).")
            return
        print(f"Trascrizione di {len(audio_files)} file da {audio_dir}...")
        str_paths = [str(p) for p in audio_files]
        result = backend.transcribe(audio=str_paths, batch_size=cfg.batch_size, channel_selector=0, verbose=True)
        transcriptions = result[0] if isinstance(result, tuple) else result
        for audio_path, text_obj in zip(audio_files, transcriptions):
            final_text = text_obj.text if hasattr(text_obj, 'text') else text_obj
            out_path = output_dir / f"{audio_path.stem}.txt"
            out_path.write_text(final_text.strip() if final_text else "", encoding='utf-8')
    return transcribe_files

def get_whisperx_transcriber(config):
    # ... (codice identico a prima, nessuna modifica)
    try:
        import whisperx
    except ImportError:
        print("\nERRORE: La libreria 'whisperx' non è installata. Attiva l'ambiente 'whisperx_env'.")
        sys.exit(1)
    cfg = config.transcription.whisperx
    print(f"Caricamento backend WhisperX con modello '{cfg.model_name}'...")
    model = whisperx.load_model(cfg.model_name, config.device, compute_type=cfg.compute_type, language=cfg.language)
    print("Backend WhisperX inizializzato.")
    def transcribe_files(audio_dir, output_dir):
        audio_files = sorted([p for p in Path(audio_dir).rglob("*.wav")])
        if not config.transcription.overwrite:
            audio_files = [f for f in audio_files if not (output_dir / f"{f.stem}.txt").exists()]
        if not audio_files:
            print(f"Nessun file da trascrivere in {audio_dir} (o già trascritti).")
            return
        print(f"Trascrizione di {len(audio_files)} file da {audio_dir}...")
        for audio_path in tqdm(audio_files, desc=f"Transcribing {audio_dir.name}"):
            try:
                audio = whisperx.load_audio(str(audio_path))
                result = model.transcribe(audio, batch_size=cfg.batch_size)
                full_text = " ".join([segment['text'].strip() for segment in result["segments"]])
                out_path = output_dir / f"{audio_path.stem}.txt"
                out_path.write_text(full_text, encoding='utf-8')
            except Exception as e:
                print(f"\nERRORE durante la trascrizione di {audio_path.name}: {e}")
                continue
    gc.collect()
    torch.cuda.empty_cache()
    return transcribe_files


def main():
    # L'import di dotenv è stato rimosso da qui.
    parser = argparse.ArgumentParser(description="Genera trascrizioni per i dati.")
    # ... (il resto della funzione main rimane identico)
    parser.add_argument("--config", type=str, required=True, help="Percorso al file config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    engine = config.transcription.engine.lower()
    transcribe_function = None
    if engine == "crisperwhisper":
        transcribe_function = get_crisperwhisper_transcriber(config)
    elif engine == "nemo":
        transcribe_function = get_nemo_transcriber(config)
    elif engine == "whisperx":
        transcribe_function = get_whisperx_transcriber(config)
    else:
        print(f"ERRORE: Engine '{engine}' non supportato. Scegli tra 'crisperwhisper', 'nemo' o 'whisperx'.")
        sys.exit(1)
    model_folder_name = ""
    if engine == "crisperwhisper":
        model_folder_name = config.transcription.crisperwhisper.model_id.split('/')[-1]
    elif engine == "nemo":
        model_folder_name = config.transcription.nemo.model_name.split('/')[-1]
    elif engine == "whisperx":
        model_folder_name = f"WhisperX_{config.transcription.whisperx.model_name}"
    train_output_dir = Path(config.data.transcripts_root) / model_folder_name
    test_output_dir = Path(config.data.test_transcripts_root) / model_folder_name
    train_output_dir.mkdir(parents=True, exist_ok=True)
    test_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Gli output verranno salvati in: {train_output_dir} e {test_output_dir}")
    print(f"\n--- Inizio Trascrizione Training Set (engine: {engine}) ---")
    transcribe_function(Path(config.data.audio_root), train_output_dir)
    print(f"\n--- Inizio Trascrizione Test Set (engine: {engine}) ---")
    transcribe_function(Path(config.data.test_audio_root), test_output_dir)
    print("\nTrascrizione completata.")

if __name__ == "__main__":
    main()