"""Where files live. Everything generated goes under ``artifacts/`` (git-ignored);
measured results that we publish go under ``results/`` (committed)."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = Path(os.environ.get("EDGEML_ARTIFACTS", ROOT / "artifacts"))
RESULTS = Path(os.environ.get("EDGEML_RESULTS", ROOT / "results"))

DATA_DIR = ARTIFACTS / "data"
PRETRAINED_DIR = ARTIFACTS / "pretrained"
RUNS_DIR = ARTIFACTS / "runs"
EXPORTS_DIR = ARTIFACTS / "exports"


def run_dir(model_name: str) -> Path:
    """Training outputs (checkpoint, tokenizer, metrics) for one model preset."""
    return RUNS_DIR / model_name


def export_path(model_name: str, precision: str) -> Path:
    return EXPORTS_DIR / f"{model_name}_{precision}.pte"
