"""The two standard CLINC150 metrics (Larson et al., 2019).

* in-scope accuracy: of the requests the assistant *should* handle, how many got the right intent.
* out-of-scope recall: of the requests it should *not* handle, how many it correctly flagged as "oos".

Reported separately because a model can look accurate overall while quietly
misrouting every out-of-scope request into some real intent.
"""

from __future__ import annotations

import torch


def intent_metrics(pred: torch.Tensor, gold: torch.Tensor, oos_id: int) -> dict[str, float]:
    pred, gold = pred.long().cpu(), gold.long().cpu()
    in_scope = gold != oos_id
    oos = ~in_scope
    return {
        "in_scope_accuracy": (pred[in_scope] == gold[in_scope]).float().mean().item() if in_scope.any() else float("nan"),
        "oos_recall": (pred[oos] == oos_id).float().mean().item() if oos.any() else float("nan"),
        "overall_accuracy": (pred == gold).float().mean().item(),
        "n_in_scope": int(in_scope.sum()),
        "n_oos": int(oos.sum()),
    }


@torch.no_grad()
def predict(model: torch.nn.Module, ids: torch.Tensor, mask: torch.Tensor, batch_size: int = 256) -> torch.Tensor:
    model.eval()
    out = [model(ids[i : i + batch_size], mask[i : i + batch_size]).argmax(-1) for i in range(0, len(ids), batch_size)]
    return torch.cat(out)
