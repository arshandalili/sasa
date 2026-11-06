from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parent


def load_config(config: str) -> dict:
    return yaml.safe_load((CONFIG_DIR / f"{config}.yaml").read_text())


def sae_config(name: str) -> dict:
    # name = "<config>/<sae>": the config's settings, then the SAE's method's, then the SAE's own
    config_name, sae_name = name.split("/")
    config = load_config(config_name)
    sae = config["saes"][sae_name]
    shared = {key: value for key, value in config.items() if key not in ("methods", "saes")}
    return {**shared, **config["methods"][sae["method"]], **sae}


def sae_names(config: str) -> list[str]:
    return [f"{config}/{sae_name}" for sae_name in load_config(config)["saes"]]
