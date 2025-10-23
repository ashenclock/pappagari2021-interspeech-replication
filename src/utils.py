import torch
import numpy as np
import random
import gc

def set_seed(seed: int):
    """Imposta il seed per la riproducibilità."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def clear_memory():
    """Libera la memoria della GPU."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()