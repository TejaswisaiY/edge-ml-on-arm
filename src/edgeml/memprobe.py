"""Memory cost of loading and running one .pte, measured in a fresh process.

Prints JSON with resident memory (RSS) right before loading the program and right after
loading it and running 20 inferences. ``delta_mib`` is what the model adds on top of the
already-imported runtime. (A peak/high-water-mark measure is not used: importing torch has
a larger transient peak than a small model, which would hide the model's cost entirely.)
Run as a subprocess by the benchmark so earlier models do not pollute the number.
"""

from __future__ import annotations

import gc
import json
import sys


def rss_mib() -> float:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    raise RuntimeError("VmRSS not found (Linux only)")


def main() -> None:
    pte, max_len = sys.argv[1], int(sys.argv[2])
    import torch

    from edgeml.runtime import PteModel

    ids = torch.ones(1, max_len, dtype=torch.long)
    mask = torch.ones(1, max_len, dtype=torch.long)
    gc.collect()
    before = rss_mib()
    m = PteModel(pte)
    for _ in range(20):
        m(ids, mask)
    gc.collect()
    after = rss_mib()
    print(json.dumps({"rss_before_mib": round(before, 2), "rss_after_mib": round(after, 2),
                      "delta_mib": round(after - before, 2)}))


if __name__ == "__main__":
    main()
