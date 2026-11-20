import pytest
import torch
from torch import nn

from edgeml.metrics import intent_metrics
from edgeml.train import linear_warmup_decay, param_groups


def test_lr_schedule_warms_up_then_decays_to_zero():
    total, warmup = 100, 10
    lrs = [linear_warmup_decay(s, total, warmup) for s in range(total + 1)]
    assert lrs[0] < lrs[5] < lrs[9] == 1.0
    assert all(a >= b for a, b in zip(lrs[warmup:], lrs[warmup + 1 :]))
    assert lrs[total] == 0.0


def test_no_weight_decay_on_bias_layernorm_embedding():
    m = nn.Sequential(nn.Embedding(5, 4), nn.Linear(4, 4), nn.LayerNorm(4))
    decay, no_decay = param_groups(m, 0.01)
    assert decay["weight_decay"] == 0.01 and no_decay["weight_decay"] == 0.0
    assert [p.shape for p in decay["params"]] == [torch.Size([4, 4])]  # only the Linear weight
    assert len(no_decay["params"]) == 4  # embedding, linear bias, LN weight, LN bias


def test_intent_metrics_separates_in_scope_and_oos():
    oos = 3
    gold = torch.tensor([0, 1, 2, 3, 3])
    pred = torch.tensor([0, 1, 0, 3, 1])
    m = intent_metrics(pred, gold, oos)
    assert m["in_scope_accuracy"] == pytest.approx(2 / 3)
    assert m["oos_recall"] == pytest.approx(0.5)
    assert m["overall_accuracy"] == pytest.approx(3 / 5)
    assert (m["n_in_scope"], m["n_oos"]) == (3, 2)
