"""Load the pretrained MiniLM (sentence-transformers/all-MiniLM-L6-v2, Apache-2.0) into our encoder.

The checkpoint is a standard BERT layout; we map its weight names onto ours and check
that every tensor is used exactly once, so a silent mismatch cannot slip through.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import torch
from safetensors.torch import load_file

from .download import fetch
from .model import EncoderConfig, IntentClassifier
from .paths import PRETRAINED_DIR

MINILM_REPO = "sentence-transformers/all-MiniLM-L6-v2"
MINILM_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
MINILM_FILES = {
    "config.json": "953f9c0d463486b10a6871cc2fd59f223b2c70184f49815e7efbcab5d8908b41",
    "tokenizer.json": "be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037",
    "model.safetensors": "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db",
}

# (pattern in the checkpoint, replacement in our model)
_RENAMES = [
    (r"^embeddings\.word_embeddings\.", "encoder.embeddings.word."),
    (r"^embeddings\.position_embeddings\.", "encoder.embeddings.position."),
    (r"^embeddings\.token_type_embeddings\.", "encoder.embeddings.token_type."),
    (r"^embeddings\.LayerNorm\.", "encoder.embeddings.norm."),
    (r"^encoder\.layer\.(\d+)\.attention\.self\.(query|key|value)\.", r"encoder.layers.\1.attn.\2."),
    (r"^encoder\.layer\.(\d+)\.attention\.output\.dense\.", r"encoder.layers.\1.attn.out."),
    (r"^encoder\.layer\.(\d+)\.attention\.output\.LayerNorm\.", r"encoder.layers.\1.attn.norm."),
    (r"^encoder\.layer\.(\d+)\.intermediate\.dense\.", r"encoder.layers.\1.ffn.up."),
    (r"^encoder\.layer\.(\d+)\.output\.dense\.", r"encoder.layers.\1.ffn.down."),
    (r"^encoder\.layer\.(\d+)\.output\.LayerNorm\.", r"encoder.layers.\1.ffn.norm."),
]
# Not used: the BERT pooler (we mean-pool instead) and a non-weight buffer.
_IGNORED = re.compile(r"^(pooler\.|embeddings\.position_ids$)")


def download_minilm(cache_dir: Path = PRETRAINED_DIR) -> Path:
    root = Path(cache_dir) / "all-MiniLM-L6-v2"
    for name, sha in MINILM_FILES.items():
        url = f"https://huggingface.co/{MINILM_REPO}/resolve/{MINILM_REVISION}/{name}"
        fetch(url, root / name, sha)
    return root


def minilm_config(root: Path, num_labels: int, max_positions: int | None = None) -> EncoderConfig:
    hf = json.loads((root / "config.json").read_text())
    if hf.get("hidden_act") != "gelu" or hf.get("position_embedding_type", "absolute") != "absolute":
        raise ValueError("unexpected MiniLM config; this loader expects a standard BERT encoder")
    return EncoderConfig(
        vocab_size=hf["vocab_size"],
        hidden=hf["hidden_size"],
        layers=hf["num_hidden_layers"],
        heads=hf["num_attention_heads"],
        intermediate=hf["intermediate_size"],
        # The checkpoint has 512 positions; we only ever use the first ``max_positions``.
        max_positions=max_positions or hf["max_position_embeddings"],
        num_labels=num_labels,
        dropout=hf["hidden_dropout_prob"],
        ln_eps=hf["layer_norm_eps"],
    )


def map_state_dict(hf_state: dict[str, torch.Tensor], max_positions: int) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key, tensor in hf_state.items():
        if _IGNORED.match(key):
            continue
        for pattern, repl in _RENAMES:
            new_key, n = re.subn(pattern, repl, key)
            if n:
                break
        else:
            raise KeyError(f"unmapped checkpoint tensor: {key}")
        if new_key == "encoder.embeddings.position.weight":
            tensor = tensor[:max_positions].clone()
        out[new_key] = tensor
    return out


def load_minilm_classifier(num_labels: int, max_positions: int, root: Path | None = None) -> IntentClassifier:
    root = root or download_minilm()
    cfg = minilm_config(root, num_labels=num_labels, max_positions=max_positions)
    model = IntentClassifier(cfg)
    state = map_state_dict(load_file(str(root / "model.safetensors")), max_positions)
    missing, unexpected = model.load_state_dict(state, strict=False)
    # Only the new classification head should be missing; nothing should be left over.
    if unexpected or set(missing) != {"classifier.weight", "classifier.bias"}:
        raise RuntimeError(f"weight loading mismatch: missing={missing} unexpected={unexpected}")
    torch.nn.init.normal_(model.classifier.weight, std=0.02)
    torch.nn.init.zeros_(model.classifier.bias)
    return model
