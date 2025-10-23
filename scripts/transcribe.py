import argparse
import sys
from pathlib import Path
from tqdm import tqdm
import torch

# Aggiunge la root del progetto al path per permettere 'from src import ...'
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config

# ===================================================================
#           LOGICA SPECIFICA E ISOLATA PER CRISPER-WHISPER
# ===================================================================
def get_crisperwhisper_transcriber(config):
    """
    Prepara e restituisce una funzione per trascrivere usando CrisperWhisper.
    Tutti gli import necessari sono DENTRO questa funzione per isolamento.
    """
    try:
        from crisper_whisper import transcribe as crisper_transcribe
    except ImportError:
        print("\nERRORE: La pipeline CrisperWhisper non è installata correttamente.")
        print("Per favore, esegui: pip install insanely-fast-whisper") # <-- CORRETTO
        sys.exit(1)
        
    print("Backend CrisperWhisper inizializzato.")
    cfg = config.transcription.crisperwhisper

    def transcribe_files(audio_dir, output_dir):
        audio_files = sorted([p for p in Path(audio_dir).rglob("*.wav")])
        if not config.transcription.overwrite:
            audio_files = [f for f in audio_files if not (output_dir / f"{f.stem}.txt").exists()]
        
        if not audio_files:
            print(f"Nessun file da trascrivere in {audio_dir} (o sono già tutti presenti).")
            return

        print(f"Trascrizione di {len(audio_files)} file da {audio_dir}...")
        for audio_path in tqdm(audio_files, desc=f"Transcribing {audio_dir.name}"):
            try:
                # La funzione 'transcribe' di CrisperWhisper fa tutto il lavoro: ASR + correzione
                result = crisper_transcribe(
                    path_to_audio_file=str(audio_path),
                    model_name=cfg.correction_model_name,
                    whisper_model_name=cfg.whisper_model_name,
                    device=config.device,
                    batch_size=cfg.batch_size,
                    compute_type=cfg.compute_type
                )
                
                out_path = output_dir / f"{audio_path.stem}.txt"
                out_path.write_text(result.text, encoding='utf-8')
            except Exception as e:
                print(f"\nERRORE durante la trascrizione di {audio_path.name}: {e}")
                continue
    
    return transcribe_files

# ===================================================================
#             LOGICA SPECIFICA E ISOLATA PER NEMO
# ===================================================================
def get_nemo_transcriber(config):
    """
    Prepara e restituisce una funzione per trascrivere usando NeMo.
    Tutti gli import necessari sono DENTRO questa funzione per isolamento.
    """
    try:
        import nemo.collections.asr as nemo_asr
    except ImportError:
        print("\nERRORE: NVIDIA NeMo Toolkit non è installato in questo ambiente.")
        print("Assicurati di essere nell'ambiente Conda corretto per NeMo.")
        sys.exit(1)

    print(f"Caricamento backend NeMo con modello '{config.transcription.model_name}'...")
    backend = nemo_asr.models.ASRModel.from_pretrained(model_name=config.transcription.model_name)
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

        result = backend.transcribe(
            paths_to_audio_files=str_paths,
            batch_size=config.transcription.batch_size,
            channel_selector='average', # Converte automaticamente stereo in mono
            verbose=False
        )
        
        # L'output di NeMo può essere una tupla (trascrizioni, logprobs)
        transcriptions = result[0] if isinstance(result, tuple) else result

        for audio_path, text_obj in zip(audio_files, transcriptions):
            final_text = ""
            if isinstance(text_obj, str):
                final_text = text_obj
            elif hasattr(text_obj, 'text'): # Gestisce l'oggetto Hypothesis
                final_text = text_obj.text
            
            out_path = output_dir / f"{audio_path.stem}.txt"
            out_path.write_text(final_text.strip() if final_text else "", encoding='utf-8')

    return transcribe_files

# ===================================================================
#                  FUNZIONE PRINCIPALE (IL "CENTRALINO")
# ===================================================================
def main():
    parser = argparse.ArgumentParser(description="Genera trascrizioni per i dati di training e test.")
    parser.add_argument("--config", type=str, default="config.yaml", required=True, help="Percorso al file di configurazione.")
    args = parser.parse_args()

    config = load_config(args.config)
    engine = config.transcription.engine.lower()

    transcribe_function = None
    # Il "centralino" decide quale funzione chiamare in base alla configurazione
    if engine == "crisperwhisper":
        transcribe_function = get_crisperwhisper_transcriber(config)
    elif engine == "nemo":
        transcribe_function = get_nemo_transcriber(config)
    else:
        print(f"ERRORE: Engine '{engine}' non supportato. Scegli tra 'crisperwhisper' o 'nemo'.")
        sys.exit(1)

    # Prepara le cartelle di output
    if engine == "crisperwhisper":
        w_model = Path(config.transcription.crisperwhisper.whisper_model_name).name
        c_model = Path(config.transcription.crisperwhisper.correction_model_name).name
        model_folder_name = f"Crisper_{w_model}" # Nome più semplice e pulito
    else: # nemo
        model_folder_name = Path(config.transcription.model_name).name

    train_output_dir = Path(config.data.transcripts_root) / model_folder_name
    test_output_dir = Path(config.data.test_transcripts_root) / model_folder_name
    
    train_output_dir.mkdir(parents=True, exist_ok=True)
    test_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Gli output verranno salvati in: {train_output_dir} e {test_output_dir}")

    # Esegue la trascrizione
    print(f"\n--- Inizio Trascrizione Training Set con engine: {engine} ---")
    transcribe_function(Path(config.data.audio_root), train_output_dir)
    
    print(f"\n--- Inizio Trascrizione Test Set con engine: {engine} ---")
    transcribe_function(Path(config.data.test_audio_root), test_output_dir)
    
    print("\nTrascrizione completata.")

if __name__ == "__main__":
    main()