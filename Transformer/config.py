from pathlib import Path
from typing import Dict, Union, Optional

def validate_config(config: Dict[str, Union[int, float, str, None]]) -> None:
    """
    Validate the configuration dictionary with comprehensive checks.
    
    Raises:
        ValueError: If any configuration parameter is invalid
    """
    # Numeric validations
    if config['batch_size'] <= 0:
        raise ValueError(f"Batch size must be positive, got {config['batch_size']}")
    
    if config['num_epochs'] <= 0:
        raise ValueError(f"Number of epochs must be positive, got {config['num_epochs']}")
    
    if config['lr'] <= 0:
        raise ValueError(f"Learning rate must be positive, got {config['lr']}")
    
    if config['seq_len'] <= 0:
        raise ValueError(f"Sequence length must be positive, got {config['seq_len']}")
    
    if config['d_model'] <= 0:
        raise ValueError(f"Model dimension must be positive, got {config['d_model']}")
    
    # Language validations
    if not config['lang_src']:
        raise ValueError("Source language cannot be empty")
    
    if not config['lang_tgt']:
        raise ValueError("Target language cannot be empty")
    
    # Path and file validations
    if not config['model_folder']:
        raise ValueError("Model folder path cannot be empty")
    
    if not config['model_basename']:
        raise ValueError("Model basename cannot be empty")
    
    # Experiment name validation
    if not config['experiment_name']:
        raise ValueError("Experiment name cannot be empty")

def get_config() -> Dict[str, Union[int, float, str, None]]:
    """
    Generate and validate the configuration dictionary.
    
    Returns:
        Dict: Validated configuration dictionary
    """
    config = {
        'batch_size': 8,
        'num_epochs': 20,
        'lr': 10**-4,
        'seq_len': 350,
        'd_model': 512,
        'lang_src': 'en',
        'lang_tgt': 'it',
        'model_folder': 'weights',
        'model_basename': 'tmodel_',
        'preload': None,
        'tokenizer_file': 'tokenizer{0}.json',
        'experiment_name': 'runs/tmodel'
    }
    
    # Validate the configuration before returning
    validate_config(config)
    return config

def get_weights(config, epochs: str) -> str:
    """
    Generate the full path for model weights.
    
    Args:
        config (Dict): Configuration dictionary
        epochs (str): Epoch number or identifier
    
    Returns:
        str: Full path to the model weights file
    """
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    model_filename = f'{model_basename}{epochs}.pt'
    return str(Path('.') / model_folder / model_filename)