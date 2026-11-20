"""Measure every (model, precision, thread count) on the current machine.

For each combination we record:
    latency  - batch size 1, fixed 32-token input, after warm-up. Median, p90, p99, mean, std.
    accuracy - of the *exported program itself* on the full test split (not of the PyTorch model),
               so a quantization bug cannot hide behind a good training score.
    size     - bytes of the .pte file.
    memory   - resident memory (RSS) added by loading + running the program, in a fresh process.

Also measured: plain PyTorch (eager) fp32, as the "no deployment tooling" baseline.

Method notes, so the numbers can be trusted and reproduced:
    * Timing uses time.perf_counter_ns around a single forward call from Python. This includes a
      few microseconds of Python/binding overhead per call, identical for every ExecuTorch variant.
    * Thread counts above the number of logical CPUs are skipped.
    * Results carry the machine description from ``sysinfo`` (CPU model, ML-relevant CPU features).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

from . import paths, sysinfo
from .data import load_clinc150
from .export import PRECISIONS, example_inputs
from .metrics import intent_metrics, predict
from .presets import MAX_LEN, PRESETS
from .runtime import PteModel, set_threads
from .tokenization import encode
from .train import load_trained


def time_calls(fn, args: tuple, warmup: int, iters: int) -> dict[str, float]:
    for _ in range(warmup):
        fn(*args)
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        fn(*args)
        samples.append((time.perf_counter_ns() - t0) / 1e6)
    samples.sort()
    q = lambda p: samples[min(len(samples) - 1, int(round(p * (len(samples) - 1))))]  # noqa: E731
    return {
        "median_ms": round(statistics.median(samples), 4),
        "p90_ms": round(q(0.90), 4),
        "p99_ms": round(q(0.99), 4),
        "mean_ms": round(statistics.fmean(samples), 4),
        "std_ms": round(statistics.pstdev(samples), 4),
        "min_ms": round(samples[0], 4),
        "iters": iters,
        "warmup": warmup,
    }


def peak_memory(pte: Path) -> dict:
    out = subprocess.run([sys.executable, "-m", "edgeml.memprobe", str(pte), str(MAX_LEN)],
                         capture_output=True, text=True, check=True, env={**os.environ, "PYTHONWARNINGS": "ignore"})
    return json.loads(out.stdout.strip().splitlines()[-1])


def benchmark_preset(preset: str, threads: list[int], warmup: int, iters: int, skip_accuracy: bool) -> list[dict]:
    model, tok, cfg = load_trained(preset)
    inputs = example_inputs(tok)
    data = load_clinc150()
    te_ids, te_mask, _ = encode(tok, data.test.texts, MAX_LEN)
    te_y = torch.tensor(data.test.labels)
    max_threads = max(threads)
    rows: list[dict] = []
    base = {"preset": preset, "parameters": cfg["parameters"]}

    # 1) PyTorch eager fp32 baseline.
    with torch.inference_mode():
        for t in threads:
            torch.set_num_threads(t)
            rows.append({**base, "backend": "pytorch_eager", "precision": "fp32", "threads": t,
                         "latency": time_calls(model, inputs, warmup, iters)})
        torch.set_num_threads(max_threads)
        if not skip_accuracy:
            acc = intent_metrics(predict(model, te_ids, te_mask), te_y, data.oos_id)
            for r in rows:
                r["test"] = acc

    # 2) ExecuTorch programs.
    for precision in PRECISIONS:
        pte = paths.export_path(preset, precision)
        if not pte.exists():
            print(f"skip {pte} (not exported)", file=sys.stderr)
            continue
        report = json.loads(pte.with_suffix(".export.json").read_text())
        group = []
        for t in threads:
            # The XNNPACK delegate binds to the threadpool when the method is loaded, so the pool is
            # resized first and the program re-loaded for every thread count (resizing under a loaded
            # method is unsafe and crashed in testing).
            set_threads(t)
            m = PteModel(pte)
            group.append({**base, "backend": "executorch_xnnpack", "precision": precision, "threads": t,
                          "pte_bytes": pte.stat().st_size,
                          "non_delegated_ops": report["not_delegated_ops"],
                          "latency": time_calls(m, inputs, warmup, iters)})
            del m
        extra = {"memory": peak_memory(pte)}
        if not skip_accuracy:
            set_threads(max_threads)
            m = PteModel(pte)
            extra["test"] = intent_metrics(m.predict(te_ids, te_mask), te_y, data.oos_id)
            del m
        for r in group:
            r.update(extra)
        rows.extend(group)
    return rows


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--presets", nargs="+", default=sorted(PRESETS), choices=sorted(PRESETS))
    ap.add_argument("--threads", nargs="+", type=int, default=[1, 2, 4])
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--skip-accuracy", action="store_true")
    ap.add_argument("--out", type=Path, help="JSON output path (default results/<machine>/benchmark.json)")
    a = ap.parse_args(argv)

    info = sysinfo.collect()
    threads = sorted({t for t in a.threads if t <= (os.cpu_count() or 1)})
    rows = []
    for preset in a.presets:
        rows.extend(benchmark_preset(preset, threads, a.warmup, a.iters, a.skip_accuracy))
        for r in rows[-(len(threads) * (1 + len(PRECISIONS))):]:
            print(f"{r['preset']:7s} {r['backend']:19s} {r['precision']:5s} t={r['threads']} "
                  f"median={r['latency']['median_ms']:.3f}ms p90={r['latency']['p90_ms']:.3f}ms", flush=True)

    out = a.out or paths.RESULTS / info["machine"] / "benchmark.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"system": info, "results": rows}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
