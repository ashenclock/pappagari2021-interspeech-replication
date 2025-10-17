import yaml
from pathlib import Path

class Config:
    def __init__(self, data: dict):
        for key, value in data.items():
            if isinstance(value, dict):
                # If the value is a dictionary, create a nested Config object
                setattr(self, key, Config(value))
            else:
                # Otherwise, just set the attribute
                setattr(self, key, value)

    def __repr__(self):
        # A simple representation for debugging
        return str(self.__dict__)

    def to_dict(self) -> dict:
        """Recursively convert the configuration object to a dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

    def copy(self, update: dict | None = None) -> "Config":
        """Create a shallow copy of the configuration with optional updates."""
        data = self.to_dict()
        update = update or {}
        data.update(update)
        return Config(data)

def load_config(config_path: str | Path) -> Config:
    """
    Loads a YAML configuration file into a nested Config object
    that allows attribute-style access.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {path.resolve()}")
        
    with open(path, 'r', encoding='utf-8') as f:
        config_data = yaml.safe_load(f)
    
    if not isinstance(config_data, dict):
        raise TypeError("The root of the YAML file must be a dictionary.")
        
    return Config(config_data)

# Example of how to use it:
# from src.config import load_config
# config = load_config()
# print(config.training.learning_rate)
