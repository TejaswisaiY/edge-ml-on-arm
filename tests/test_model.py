import pytest
import torch

from edgeml.model import EncoderConfig, IntentClassifier, count_parameters, init_weights
from edgeml.presets import MAX_LEN, PRESETS


def small_model(**kw) -> IntentClassifier:
    cfg = EncoderConfig(**{"vocab_size": 100, "hidden": 32, "layers": 2, "heads": 4, "intermediate": 64,
                           "max_positions": 16, "num_labels": 7, **kw})
    torch.manual_seed(0)
    m = IntentClassifier(cfg)
    init_weights(m)
    return m.eval()


def test_output_shape():
    m = small_model()
    ids = torch.randint(4, 100, (3, 16))
    mask = torch.ones_like(ids)
    assert m(ids, mask).shape == (3, 7)


def test_padding_does_not_change_prediction():
    """Whatever sits in padded positions must not affect the result (attention + pooling are masked)."""
    m = small_model()
    ids = torch.randint(4, 100, (1, 16))
    mask = torch.zeros_like(ids)
    mask[:, :6] = 1
    other = ids.clone()
    other[:, 6:] = torch.randint(4, 100, (1, 10))
    with torch.no_grad():
        torch.testing.assert_close(m(ids, mask), m(other, mask), atol=1e-6, rtol=0)


def test_heads_must_divide_hidden():
    with pytest.raises(ValueError):
        EncoderConfig(vocab_size=10, hidden=30, layers=1, heads=4, intermediate=8, max_positions=4, num_labels=2)


def test_tiny_preset_is_about_one_million_parameters():
    p = PRESETS["tiny"]
    cfg = EncoderConfig(vocab_size=p.vocab_size, hidden=p.hidden, layers=p.layers, heads=p.heads,
                        intermediate=p.intermediate, max_positions=MAX_LEN, num_labels=151)
    counts = count_parameters(IntentClassifier(cfg))
    assert 0.8e6 < counts["total"] < 1.1e6
    assert counts["embeddings"] + counts["non_embedding"] == counts["total"]
