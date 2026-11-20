"""Small hyper-parameter sweep for the tiny model, decided on the VALIDATION split only.

The test split is not evaluated here (evaluate_test=False). The winning settings are then
written into presets.py and the final model is trained once, and only then scored on test.

Usage: python scripts/sweep_tiny.py [--threads N]
"""

import argparse
import json
from pathlib import Path

from edgeml import paths
from edgeml.train import train

GRID = [
    {"epochs": 15, "dropout": 0.1, "label_smoothing": 0.0},
    {"epochs": 15, "dropout": 0.2, "label_smoothing": 0.1},
    {"epochs": 15, "dropout": 0.3, "label_smoothing": 0.1},
    {"epochs": 25, "dropout": 0.3, "label_smoothing": 0.1},
]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int)
    a = ap.parse_args()
    out = paths.RESULTS / "sweeps" / "tiny_val.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for cfg in GRID:
        name = "tiny_" + "_".join(f"{k}{v}" for k, v in cfg.items())
        m = train("tiny", paths.RUNS_DIR / "sweep" / name, threads=a.threads, overrides=dict(cfg), evaluate_test=False)
        rows.append({"settings": cfg, "best_epoch": m["best_epoch"], "val": m["val"]})
        out.write_text(json.dumps(rows, indent=2))
    best = max(rows, key=lambda r: r["val"]["overall_accuracy"])
    print("best on validation:", json.dumps(best))
