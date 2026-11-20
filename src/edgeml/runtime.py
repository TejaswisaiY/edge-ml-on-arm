"""Thin wrapper around the ExecuTorch runtime (the same C++ runtime used on phones, via Python bindings)."""

from __future__ import annotations

from pathlib import Path

import torch
from executorch.extension.pybindings import portable_lib
from executorch.runtime import Runtime


def set_threads(n: int) -> None:
    """Resize ExecuTorch's CPU threadpool (used by the XNNPACK kernels)."""
    portable_lib._unsafe_reset_threadpool(n)


def get_threads() -> int:
    return portable_lib._threadpool_get_thread_count()


class PteModel:
    """Callable like the PyTorch model: ``logits = m(input_ids, attention_mask)``."""

    def __init__(self, path: Path, enable_etdump: bool = False, debug_buffer_size: int = 0) -> None:
        self.path = Path(path)
        self.program = Runtime.get().load_program(
            self.path, enable_etdump=enable_etdump, debug_buffer_size=debug_buffer_size
        )
        method = self.program.load_method("forward")
        if method is None:
            raise RuntimeError(f"{path} has no 'forward' method")
        self.method = method

    def __call__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.method.execute([input_ids, attention_mask])[0]

    def predict(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Classify row by row (batch size 1, as on a device)."""
        return torch.stack([self(ids[i : i + 1], mask[i : i + 1]).argmax(-1)[0] for i in range(len(ids))])
