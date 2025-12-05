"""
Configuration Loader
-------------------
Load and parse YAML/JSON configuration files.
Centralized settings management for the project.

Usage:
    from src.utils.config_loader import load_config, get_value
    
    config = load_config('configs/config.yaml')
    lr = config['training']['learning_rate']
    
    # Or use helper
    lr = get_value(config, 'training.learning_rate', default=0.001)
"""

import os
import yaml
import json
from pathlib import Path
from src.utils.logger import get_logger
# from src.utils.logger import setup_logger # for standalone file testing

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LOGS_DIR = os.path.join(PROJECT_ROOT, "outputs", "logs")
# Setup logger
# logger = setup_logger(__name__, os.path.join(LOGS_DIR, 'config_loader.log')) # for standalone file testing
logger = get_logger(
    __name__
)

CONFIGS_DIR= os.path.join(PROJECT_ROOT, "configs")
DEFAULT_CONFIG_PATH = os.path.join(CONFIGS_DIR, 'config.yaml')

def load_config(config_path=DEFAULT_CONFIG_PATH):
    """
    Load configuration from YAML or JSON file
    
    Args:
        config_path (str): Path to config file (default: configs/config.yaml)
        
    Returns:
        dict: Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If unsupported file format
        
    Example:
        # Load default config
        config = load_config()
        
        # Load specific config
        config = load_config('configs/experiment1.yaml')
    """
   
    logger.info("="*60)
    logger.info("Loading Configuration")
    logger.info("="*60)
    
    config_path = Path(config_path)
    
    # Check if file exists
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    logger.info(f"Loading config from: {config_path}...")
    
    # Load based on file extension
    try:
        if config_path.suffix in ['.yaml', '.yml']:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info("Config loaded from YAML file")
            
        elif config_path.suffix == '.json':
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info("Config loaded from JSON file")
            
        else:
            logger.error(f"Unsupported file format: {config_path.suffix}")
            raise ValueError(f"Unsupported config format: {config_path.suffix}. Use .yaml, .yml, or .json")
        
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        raise
    
    # Validate config
    if config is None or not isinstance(config, dict):
        logger.error("Invalid config format: must be a dictionary")
        raise ValueError("Config file is empty or invalid")
    
    logger.info(f"Config sections: {list(config.keys())}")
    logger.info("="*60)
    
    return config


def get_value(config, key_path, default=None):
    """
    Get nested config value using dot notation
    
    Args:
        config (dict): Configuration dictionary
        key_path (str): Dot-separated path (e.g., 'training.learning_rate')
        default: Default value if key not found
        
    Returns:
        Value at key_path or default
        
    Example:
        lr = get_value(config, 'training.learning_rate', default=0.001)
        # Instead of: config['training']['learning_rate']
    """
    
    keys = key_path.split('.')
    value = config
    
    try:
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        logger.warning(f"Key not found: {key_path}, using default: {default}")
        return default


def save_config(config, save_path):
    """
    Save configuration to YAML file
    
    Args:
        config (dict): Configuration dictionary
        save_path (str): Path to save config
        
    Example:
        config = {'training': {'lr': 0.001}}
        save_config(config, 'configs/experiment1.yaml')
    """
    
    logger.info(f"Saving config to: {save_path}")
    
    save_path = Path(save_path)
    
    # Create directory if needed
    os.makedirs(save_path.parent, exist_ok=True)
    
    # Save based on extension
    try:
        if save_path.suffix in ['.yaml', '.yml']:
            with open(save_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            logger.info("Config saved as YAML")
            
        elif save_path.suffix == '.json':
            with open(save_path, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info("Config saved as JSON")
            
        else:
            logger.error(f"Unsupported format: {save_path.suffix}")
            raise ValueError(f"Use .yaml, .yml, or .json extension")
        
        logger.info("Config saved successfully")
        
    except Exception as e:
        logger.error(f"Error saving config: {str(e)}")
        raise


def validate_config(config, required_keys):
    """
    Validate that config has required keys
    
    Args:
        config (dict): Configuration to validate
        required_keys (list): List of required key paths (dot notation)
        
    Returns:
        tuple: (is_valid, missing_keys)
        
    Example:
        required = ['training.batch_size', 'training.learning_rate']
        valid, missing = validate_config(config, required)
        if not valid:
            print(f"Missing keys: {missing}")
    """
    
    missing_keys = []
    
    for key_path in required_keys:
        value = get_value(config, key_path, default='__MISSING__')
        if value == '__MISSING__':
            missing_keys.append(key_path)
    
    is_valid = len(missing_keys) == 0
    
    if is_valid:
        logger.info("Config validation passed")
    else:
        logger.warning(f"Config validation failed. Missing keys: {missing_keys}")
    
    return is_valid, missing_keys


def merge_configs(base_config, override_config):
    """
    Merge two configs (override_config takes precedence)
    
    Args:
        base_config (dict): Base configuration
        override_config (dict): Override configuration
        
    Returns:
        dict: Merged configuration
        
    Example:
        base = load_config('configs/base.yaml')
        exp = load_config('configs/experiment.yaml')
        config = merge_configs(base, exp)
    """
    
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            # Recursive merge for nested dicts
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    logger.info("Configs merged successfully")
    
    return merged


def create_default_config():
    """
    Create default configuration template
    
    Returns:
        dict: Default configuration
        
    Example:
        config = create_default_config()
        save_config(config, 'configs/config.yaml')
    """
    
    default_config = {
        'model': {
            'name': 'resnet18',
            'pretrained': True,
            'num_classes': 3,
            'freeze_backbone': False
        },
        
        'data': {
            'train_dir': 'data/water_dataset_split/train',
            'test_dir': 'data/water_dataset_split/test',
            'image_size': 224,
            'batch_size': 32,
            'num_workers': 4
        },
        
        'training': {
            'epochs': 50,
            'learning_rate': 0.001,
            'optimizer': 'adam',
            'weight_decay': 0.0001,
            'scheduler': 'step',
            'step_size': 10,
            'gamma': 0.1
        },
        
        'augmentation': {
            'horizontal_flip': True,
            'rotation': 15,
            'brightness': 0.2,
            'contrast': 0.2
        },
        
        'paths': {
            'save_dir': 'models/resnet18',
            'log_dir': 'outputs/logs',
            'checkpoint_dir': 'models/resnet18/checkpoints'
        },
        
        'logging': {
            'save_frequency': 5,
            'log_interval': 10
        }
    }
    
    logger.info("Default config created")
    
    return default_config


# Testing
if __name__ == "__main__":
    """
    Test config loader functions
    Run: python src/utils/config_loader.py
    """
    
    logger.info("="*60)
    logger.info("Testing Config Loader")
    logger.info("="*60)
    
    # Test 1: Create default config
    logger.info("1. Creating default config...")
    try:
        config = create_default_config()
        logger.info(f"Config sections: {list(config.keys())}")
        logger.info("Default config created!")
    except Exception as e:
        logger.error(f"Error: {str(e)}")
    
    # Test 2: Save config
    logger.info("2. Saving config...")
    try:
        test_path = 'configs/test_config.yaml'
        save_config(config, test_path)
        logger.info(f"Config saved to: {test_path}")
    except Exception as e:
        logger.error(f"Error: {str(e)}")
    
    # Test 3: Load config
    logger.info("3. Loading config...")
    try:
        loaded_config = load_config('configs/test_config.yaml')
        logger.info(f"Config loaded with {len(loaded_config)} sections")
    except Exception as e:
        logger.error(f"Error: {str(e)}")
    
    # Test 4: Get values
    logger.info("4. Testing get_value...")
    try:
        lr = get_value(config, 'training.learning_rate')
        batch = get_value(config, 'data.batch_size')
        missing = get_value(config, 'missing.mykey', default='myvalue')
        
        logger.info(f"Learning rate: {lr}")
        logger.info(f"Batch size: {batch}")
        logger.info(f"Missing key: {missing}")
    except Exception as e:
        logger.error(f"Error: {str(e)}")
    
    # Test 5: Validate config
    logger.info("5. Testing validation...")
    try:
        required = [
            'training.learning_rate',
            'training.epochs',
            'data.batch_size'
        ]
        valid, missing = validate_config(config, required)
        
        if valid:
            logger.info(f"All required key {required[0]}, {required[1]}, {required[2]} found. Validation passed!")
        else:
            logger.warning(f"Missing keys: {missing}")
    except Exception as e:
        logger.error(f"Error: {str(e)}")
    
    # Test 6: Merge configs
    logger.info("6. Testing merge...")
    try:
        override = {'training': {'learning_rate': 0.0001}}
        merged = merge_configs(config, override)
        
        logger.info(f"Original LR: {config['training']['learning_rate']}")
        logger.info(f"Merged LR: {merged['training']['learning_rate']}")
    except Exception as e:
        logger.error(f"Error: {str(e)}")
    
    # Cleanup
    logger.info("7. Cleaning up...")
    try:
        if os.path.exists('configs/test_config.yaml'):
            os.remove('configs/test_config.yaml')
            logger.info("Test file removed")
    except Exception as e:
        logger.error(f"Error: {str(e)}")
    
    logger.info("="*60)
    logger.info("Config loader tests complete!")
    logger.info("="*60)