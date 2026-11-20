"""Record exactly what machine a measurement came from. A latency number without this is meaningless."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import re
import subprocess
import sys


# CPU features that matter for int8/fp16 ML kernels.
ARM_FEATURES = ("asimd", "asimddp", "i8mm", "bf16", "fphp", "asimdhp", "sve", "sve2", "sme")
X86_FEATURES = ("avx2", "fma", "avx512f", "avx512_vnni", "avx_vnni", "avx512_bf16", "amx_int8")


def _lscpu() -> dict[str, str]:
    try:
        out = subprocess.run(["lscpu"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    return {k.strip(): v.strip() for k, v in (line.split(":", 1) for line in out.splitlines() if ":" in line)}


def _cpu_flags() -> set[str]:
    try:
        text = open("/proc/cpuinfo").read()
    except OSError:
        return set()
    m = re.search(r"^(?:flags|Features)\s*:\s*(.*)$", text, re.MULTILINE)
    return set(m.group(1).split()) if m else set()


def collect() -> dict:
    import torch

    lscpu = _lscpu()
    flags = _cpu_flags()
    machine = platform.machine()
    wanted = ARM_FEATURES if machine in ("aarch64", "arm64") else X86_FEATURES
    info = {
        "machine": machine,
        "cpu_model": lscpu.get("Model name") or platform.processor() or "unknown",
        "cpu_vendor": lscpu.get("Vendor ID"),
        "logical_cpus": os.cpu_count(),
        "ml_cpu_features": {f: (f in flags) for f in wanted},
        "l2_cache": lscpu.get("L2 cache"),
        "l3_cache": lscpu.get("L3 cache"),
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "executorch": importlib.metadata.version("executorch"),
    }
    if os.environ.get("GITHUB_RUN_ID"):
        info["ci"] = {
            "runner_label": os.environ.get("EDGEML_RUNNER_LABEL"),
            "run_url": f"{os.environ.get('GITHUB_SERVER_URL')}/{os.environ.get('GITHUB_REPOSITORY')}"
                       f"/actions/runs/{os.environ['GITHUB_RUN_ID']}",
            "commit": os.environ.get("GITHUB_SHA"),
        }
    return info


if __name__ == "__main__":
    import json

    print(json.dumps(collect(), indent=2))
