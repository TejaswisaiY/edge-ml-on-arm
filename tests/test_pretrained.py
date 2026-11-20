"""Our hand-written encoder must reproduce the reference BERT implementation exactly."""

import pytest
import torch

from edgeml.pretrained import _IGNORED, download_minilm, load_minilm_classifier, map_state_dict
from edgeml.tokenization import encode, load_tokenizer

pytestmark = [pytest.mark.network, pytest.mark.slow]


def test_every_checkpoint_tensor_is_mapped():
    from safetensors.torch import load_file

    root = download_minilm()
    hf = load_file(str(root / "model.safetensors"))
    mapped = map_state_dict(hf, max_positions=32)
    used = [k for k in hf if not _IGNORED.match(k)]
    assert len(mapped) == len(used) == 5 + 6 * 16  # embeddings + 6 layers x 16 tensors


def test_matches_reference_bert():
    transformers = pytest.importorskip("transformers")
    root = download_minilm()
    ref = transformers.BertModel.from_pretrained(str(root), add_pooling_layer=False).eval()
    ours = load_minilm_classifier(num_labels=151, max_positions=32).eval()
    tok = load_tokenizer(root / "tokenizer.json")
    ids, mask, _ = encode(tok, ["remind me to call amma at 6", "what's the weather like in lund"], 32)
    with torch.no_grad():
        a = ref(input_ids=ids, attention_mask=mask).last_hidden_state
        b = ours.encoder(ids, mask)
    real = mask.bool()
    torch.testing.assert_close(b[real], a[real], atol=1e-4, rtol=1e-4)
