"""Turn a trained PyTorch model into an ExecuTorch program (``.pte``) that runs on an Arm CPU.

Pipeline:
    PyTorch model
      -> torch.export            (capture the graph with fixed input shapes)
      -> [optional] PT2E quantization (insert int8 quantize/dequantize around Linear layers)
      -> to_edge + XNNPACK partitioner (hand supported ops to XNNPACK, Arm-optimised CPU kernels)
      -> to_executorch           (serialise to a .pte file the C++ runtime can load)

Precisions:
    fp32  - weights and activations in 32-bit float.
    int8  - Linear weights stored as int8 (one scale per output channel); activations are
            quantized to int8 on the fly for each matmul ("dynamic" quantization). Embedding
            tables, LayerNorm, softmax and the attention matmuls stay fp32.

Each export also records which operators XNNPACK accepted ("delegated") and which fell back
to ExecuTorch's portable kernels, because a fallback op can quietly dominate latency.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import XNNPACKQuantizer, get_symmetric_quantization_config
from executorch.devtools.backend_debug import get_delegation_info
from executorch.exir import to_edge_transform_and_lower
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

from . import paths
from .presets import MAX_LEN, PRESETS
from .tokenization import encode

PRECISIONS = ("fp32", "int8")
EXAMPLE_SENTENCE = "remind me to call my mum at six tonight"


def example_inputs(tok, max_len: int = MAX_LEN) -> tuple[torch.Tensor, torch.Tensor]:
    ids, mask, _ = encode(tok, [EXAMPLE_SENTENCE], max_len)
    return ids, mask


def int8_dynamic_config():
    return get_symmetric_quantization_config(is_per_channel=True, is_dynamic=True)


def quantize(model: torch.nn.Module, inputs: tuple, module_names: list[str] | None = None) -> torch.nn.Module:
    """Apply int8 dynamic quantization to all Linear layers, or only to ``module_names``."""
    quantizer = XNNPACKQuantizer()
    if module_names is None:
        quantizer.set_global(int8_dynamic_config())
    else:
        for name in module_names:
            quantizer.set_module_name(name, int8_dynamic_config())
    gm = torch.export.export(model, inputs, strict=True).module()
    gm = prepare_pt2e(gm, quantizer)
    gm(*inputs)  # dynamic quant needs no calibration data; one pass finalises observers
    return convert_pt2e(gm)


def delegation_report(edge_gm: torch.fx.GraphModule) -> dict:
    info = get_delegation_info(edge_gm)
    by_op = {
        op: {"delegated": b.delegated, "not_delegated": b.non_delegated}
        for op, b in sorted(info.delegation_by_operator.items())
    }
    return {
        "num_delegated_subgraphs": info.num_delegated_subgraphs,
        "num_delegated_nodes": info.num_delegated_nodes,
        "num_non_delegated_nodes": info.num_non_delegated_nodes,
        "not_delegated_ops": {op: v["not_delegated"] for op, v in by_op.items() if v["not_delegated"]},
        "by_operator": by_op,
    }


def export_program(model: torch.nn.Module, inputs: tuple, out_path: Path, precision: str,
                   int8_modules: list[str] | None = None) -> dict:
    """Export ``model`` to ``out_path`` and return a small report."""
    if precision not in (*PRECISIONS, "mixed"):
        raise ValueError(f"unknown precision {precision!r}")
    t0 = time.perf_counter()
    model = model.eval()
    with torch.no_grad():
        if precision == "int8":
            module = quantize(model, inputs)
        elif precision == "mixed":
            module = quantize(model, inputs, module_names=int8_modules or [])
        else:
            module = model
        ep = torch.export.export(module, inputs, strict=True)
        edge = to_edge_transform_and_lower(ep, partitioner=[XnnpackPartitioner()])
        report = delegation_report(edge.exported_program().graph_module)
        program = edge.to_executorch()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(program.buffer)
    report.update({
        "precision": precision,
        "int8_modules": int8_modules if precision == "mixed" else None,
        "pte_bytes": out_path.stat().st_size,
        "export_seconds": round(time.perf_counter() - t0, 2),
        "input_shape": list(inputs[0].shape),
    })
    out_path.with_suffix(".export.json").write_text(json.dumps(report, indent=2))
    return report


def main(argv: list[str] | None = None) -> None:
    from .train import load_trained

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("preset", choices=sorted(PRESETS))
    ap.add_argument("--precision", choices=PRECISIONS, nargs="+", default=list(PRECISIONS))
    ap.add_argument("--run-dir", type=Path)
    a = ap.parse_args(argv)
    model, tok, _ = load_trained(a.preset, a.run_dir)
    inputs = example_inputs(tok)
    for precision in a.precision:
        rep = export_program(model, inputs, paths.export_path(a.preset, precision), precision)
        print(json.dumps({k: rep[k] for k in ("precision", "pte_bytes", "export_seconds", "num_non_delegated_nodes",
                                               "not_delegated_ops")}), flush=True)


if __name__ == "__main__":
    main()
