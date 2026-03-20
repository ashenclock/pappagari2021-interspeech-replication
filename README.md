# Replication: Pappagari et al., Interspeech 2021

Replication of *"Automatic Detection and Assessment of Alzheimer's Disease Using Speech and Language Technologies in Low-Resource Scenarios"* (Pappagari et al., Interspeech 2021).

This project implements a pipeline for Alzheimer's Disease detection using both **deep learning** (BERT for text, ECAPA-TDNN for audio) and **classical ML** (SVM, XGBoost, Logistic Regression) approaches with cross-validation and model fusion.

## Project Structure

```
.
├── config.yaml                         # Main configuration file
├── requirements.txt                    # Python dependencies
├── src/
│   ├── config.py                       # YAML config loader with dot notation
│   ├── data.py                         # Datasets, dataloaders, K-Fold splits
│   ├── engine.py                       # Deep learning trainer, predictor, evaluator
│   ├── models.py                       # BertClassifier, EcapaTdnnClassifier
│   ├── tabular_engine.py               # Tabular ML trainer/predictor (SVM, XGBoost, LR)
│   └── utils.py                        # Seed setting, memory management
├── scripts/
│   ├── run.py                          # Main script: deep learning train/predict/evaluate
│   ├── run_tabular.py                  # Tabular ML train/predict/evaluate
│   ├── transcribe.py                   # ASR transcription (WhisperX, NeMo, CrisperWhisper)
│   ├── extract_features.py             # Acoustic feature extraction (openSMILE, MFCC, Whisper embeddings)
│   ├── extract_embeddings_batch.py     # GPU-batched Whisper embedding extraction
│   ├── fuse_models.py                  # Late fusion with grid search
│   └── generate_scores.py             # Generate OOF and test scores for fusion
```

## Pipeline

### 1. Transcription (ASR)
```bash
python scripts/transcribe.py --config config.yaml
```
Supports three ASR backends: **WhisperX**, **NeMo Parakeet**, and **CrisperWhisper**. Configure the engine in `config.yaml` under `transcription.engine`.

### 2. Feature Extraction
```bash
# Acoustic features (eGeMAPS, ComParE, MFCC, Disvoice, Whisper embeddings)
python scripts/extract_features.py --config config.yaml

# Or GPU-batched Whisper embeddings (faster)
python scripts/extract_embeddings_batch.py --config config.yaml
```

### 3a. Deep Learning Training & Evaluation
```bash
# Train (K-Fold cross-validation)
python scripts/run.py --config config.yaml --mode train

# Predict on test set
python scripts/run.py --config config.yaml --mode predict

# Evaluate predictions
python scripts/run.py --config config.yaml --mode evaluate
```

### 3b. Tabular ML Training & Evaluation
```bash
python scripts/run_tabular.py --config config.yaml --mode train
python scripts/run_tabular.py --config config.yaml --mode predict
python scripts/run_tabular.py --config config.yaml --mode evaluate
```

### 4. Model Fusion (optional)
```bash
# Generate OOF scores from trained models
python scripts/generate_scores.py --config config.yaml

# Late fusion with grid search
python scripts/fuse_models.py --config config.yaml --evaluate
```

## Configuration

All parameters are controlled via `config.yaml`:
- **Data paths**: audio, transcripts, features, labels
- **ASR engine**: WhisperX, NeMo, or CrisperWhisper
- **Feature extraction**: eGeMAPS, ComParE, MFCC, Disvoice, Whisper embeddings
- **Task**: `classification` (AD vs CN) or `regression` (MMSE prediction)
- **Modality**: `text` (BERT) or `audio` (ECAPA-TDNN)
- **Training**: epochs, learning rate, early stopping, K-Fold CV

## Setup

```bash
pip install -r requirements.txt
```

For ASR backends, install the specific dependency you need (see comments in `requirements.txt`).

## Reference

```bibtex
@inproceedings{pappagari2021,
  title={Automatic Detection and Assessment of Alzheimer Disease Using Speech and Language Technologies in Low-Resource Scenarios},
  author={Pappagari, Raghavendra and Cho, Jaejin and Joshi, Sonal and Moro-Vel{\'a}zquez, Laureano and Dehak, Najim},
  booktitle={Proc. Interspeech},
  year={2021}
}
```
