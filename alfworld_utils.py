import os
import random
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import yaml


def apply_textworld_py313_compat() -> None:
    """Patch TextWorld eval context handling for Python 3.13+."""
    if sys.version_info < (3, 13):
        return

    from textworld.envs.pddl.textgen import EvalSymbol, TerminalSymbol

    def _derive(self, context=None):
        context = context or self.context
        variables = context.get("variables", {})
        value = eval(self.expression, {}, variables)
        return [TerminalSymbol(value)]

    EvalSymbol.derive = _derive


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def expand_env_vars(obj):
    if isinstance(obj, dict):
        return {k: expand_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [expand_env_vars(v) for v in obj]
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    return obj


def load_config(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return expand_env_vars(config)


def map_split_name(split: str) -> str:
    if split == "eval_id":
        return "eval_in_distribution"
    if split == "eval_ood":
        return "eval_out_of_distribution"
    raise ValueError(f"Unsupported split: {split}")


def validate_config_paths(config: Dict) -> None:
    required_paths = [
        ("dataset.data_path", config["dataset"]["data_path"]),
        ("dataset.eval_id_data_path", config["dataset"]["eval_id_data_path"]),
        ("dataset.eval_ood_data_path", config["dataset"]["eval_ood_data_path"]),
        ("logic.domain", config["logic"]["domain"]),
        ("logic.grammar", config["logic"]["grammar"]),
    ]
    missing = [(name, path) for name, path in required_paths if not Path(path).exists()]
    if missing:
        details = "\n".join([f"- {name}: {path}" for name, path in missing])
        raise FileNotFoundError(
            "ALFWorld data paths are missing. Set ALFWORLD_DATA correctly.\n"
            f"Missing paths:\n{details}"
        )
