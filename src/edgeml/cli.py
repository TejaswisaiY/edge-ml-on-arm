"""``edgeml <command> ...`` - one entry point for the whole pipeline.

    edgeml train tiny|minilm        train a model (plain PyTorch loop)
    edgeml export tiny|minilm       export to ExecuTorch .pte (fp32, int8)
    edgeml benchmark                measure latency / size / memory / accuracy on this machine
    edgeml report                   regenerate results/SUMMARY.md
    edgeml sysinfo                  describe this machine
"""

from __future__ import annotations

import importlib
import sys

COMMANDS = {"train": "train", "export": "export", "benchmark": "benchmark", "report": "report", "sysinfo": "sysinfo"}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        sys.exit(2)
    module = importlib.import_module(f"edgeml.{COMMANDS[sys.argv[1]]}")
    if sys.argv[1] == "sysinfo":
        import json

        print(json.dumps(module.collect(), indent=2))
        return
    module.main(sys.argv[2:])
