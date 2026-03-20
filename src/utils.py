import torch
import numpy as np
import random
import gc


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --- Shared label utilities ---

DIAGNOSIS_COLUMNS = ['label', 'dx', 'diagnosis', 'Dx', 'DISEASE']

LABEL_MAPPING = {
    'Control': 0, 'ProbableAD': 1,
    'CN': 0, 'AD': 1, 'MCI': 0, 'CTR': 0,
    'cn': 0, 'ad': 1, 'mci': 0,
}


def find_diagnosis_column(df):
    """Find the diagnosis column in a DataFrame, or None."""
    return next((c for c in DIAGNOSIS_COLUMNS if c in df.columns), None)


def map_label(val):
    """Map a diagnosis string to binary label (AD=1, else=0)."""
    s = str(val).upper().strip()
    if 'AD' in s:
        return 1
    return 0


def compute_classification_metrics(y_true, y_pred):
    """Compute accuracy, F1, sensitivity, specificity from true/pred arrays."""
    from sklearn.metrics import (
        accuracy_score, f1_score, confusion_matrix, classification_report
    )
    accuracy = accuracy_score(y_true, y_pred)
    f1_w = f1_score(y_true, y_pred, average='weighted')
    f1_macro = f1_score(y_true, y_pred, average='macro')
    cm = confusion_matrix(y_true, y_pred)

    sensitivity, specificity = 0, 0
    if len(cm.ravel()) == 4:
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    report = classification_report(y_true, y_pred, target_names=['CN', 'AD'])

    return {
        'accuracy': accuracy,
        'f1_weighted': f1_w,
        'f1_macro': f1_macro,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'confusion_matrix': cm,
        'classification_report': report,
    }


def load_audio(path, target_sr=16000):
    """Load audio file, convert to mono, resample if needed. Returns (waveform_1d, sr)."""
    import torchaudio
    from torchaudio.functional import resample
    waveform, sr = torchaudio.load(str(path))
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    if sr != target_sr:
        waveform = resample(waveform, sr, target_sr)
    return waveform.squeeze(0), target_sr


def load_labels_df(path, id_column='ID'):
    """Load a CSV label file, normalizing the ID column name."""
    import pandas as pd
    df = pd.read_csv(path)
    if 'adressfname' in df.columns:
        df = df.rename(columns={'adressfname': id_column})
    # Also check for generic id-like columns
    if id_column not in df.columns:
        id_col = next((c for c in df.columns if 'id' in c.lower() or 'name' in c.lower()), None)
        if id_col:
            df = df.rename(columns={id_col: id_column})
    df[id_column] = df[id_column].astype(str)
    return df
