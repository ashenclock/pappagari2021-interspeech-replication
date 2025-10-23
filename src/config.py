import yaml
from pathlib import Path
from typing import Any

class Config:
    """
    Classe per caricare la configurazione da un file YAML e permettere
    l'accesso agli attributi tramite dot notation (es. config.data.audio_root).
    """
    def __init__(self, data: dict):
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    def __repr__(self):
        return str(self.__dict__)

    def to_dict(self) -> dict:
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

def load_config(config_path: str | Path) -> Config:
    """Carica un file di configurazione YAML."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"File di configurazione non trovato in: {path.resolve()}")
        
    with open(path, 'r', encoding='utf-8') as f:
        config_data = yaml.safe_load(f)
    
    if not isinstance(config_data, dict):
        raise TypeError("La radice del file YAML deve essere un dizionario.")
        
    return Config(config_data)