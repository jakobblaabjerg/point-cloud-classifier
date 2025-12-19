import yaml
import os

def load_config(base_path, specific_path=None):
    with open(base_path) as f:
        base_config = yaml.safe_load(f)

    if specific_path:
        with open(specific_path) as f:
            specific_config = yaml.safe_load(f)
        if specific_config and isinstance(specific_config, dict):
            config = merge_dicts(base_config, specific_config)
        else:
            config = base_config
    else:
        config = base_config
    return config


def merge_dicts(base, specific):
    """Recursively merge specific into base."""
    for key, value in specific.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            merge_dicts(base[key], value)
        else:
            base[key] = value
    return base


def save_config(config, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    config_path = os.path.join(log_dir, "config.yaml")
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f)