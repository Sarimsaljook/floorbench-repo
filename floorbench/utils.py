from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from floorbench.config import RESULTS_ROOT


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def enable_determinism() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s | %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


class ResultWriter:
    """Owns the on-disk layout for one experiment."""

    def __init__(self, experiment: str, root: Path = RESULTS_ROOT) -> None:
        self.root = root / experiment
        self.figures = self.root / "figures"
        self.tables = self.root / "tables"
        self.raw = self.root / "raw"
        for directory in (self.figures, self.tables, self.raw):
            directory.mkdir(parents=True, exist_ok=True)

    def write_table(self, frame: pd.DataFrame, name: str) -> Path:
        path = self.tables / f"{name}.csv"
        frame.to_csv(path, index=False)
        return path

    def write_json(self, payload: dict[str, Any], name: str) -> Path:
        path = self.raw / f"{name}.json"
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=_json_default)
        return path

    def figure_path(self, name: str, suffix: str = "png") -> Path:
        return self.figures / f"{name}.{suffix}"


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"unserialisable type {type(value)!r}")
