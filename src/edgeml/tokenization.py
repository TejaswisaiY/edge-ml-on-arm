"""Tokenizers for both models.

* The tiny model gets its own small WordPiece vocabulary, learned from the training split only.
  A small vocabulary is part of being small: the embedding table is ``vocab x hidden`` and
  would otherwise dominate the model size.
* The pretrained MiniLM uses the exact tokenizer it was trained with.

Both produce fixed-length inputs (``max_len``), because a fixed shape is what lets the
exported program be planned ahead of time on the device.
"""

from __future__ import annotations

from pathlib import Path

import torch
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, processors, trainers

PAD, UNK, CLS, SEP = "[PAD]", "[UNK]", "[CLS]", "[SEP]"


def train_wordpiece(texts: list[str], vocab_size: int) -> Tokenizer:
    tok = Tokenizer(models.WordPiece(unk_token=UNK))
    tok.normalizer = normalizers.BertNormalizer(lowercase=True)
    tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tok.decoder = decoders.WordPiece()
    trainer = trainers.WordPieceTrainer(vocab_size=vocab_size, special_tokens=[PAD, UNK, CLS, SEP], min_frequency=1)
    tok.train_from_iterator(texts, trainer=trainer)
    tok.post_processor = processors.TemplateProcessing(
        single=f"{CLS} $A {SEP}", special_tokens=[(CLS, tok.token_to_id(CLS)), (SEP, tok.token_to_id(SEP))]
    )
    return tok


def load_tokenizer(path: Path) -> Tokenizer:
    return Tokenizer.from_file(str(path))


def encode(tok: Tokenizer, texts: list[str], max_len: int) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Return ``(input_ids, attention_mask, truncated_fraction)``, both tensors ``[N, max_len]`` int64."""
    pad_id = tok.token_to_id(PAD)
    if pad_id is None:
        raise ValueError("tokenizer has no [PAD] token")
    tok.no_truncation()
    tok.no_padding()
    raw = tok.encode_batch(texts)
    truncated = sum(len(e.ids) > max_len for e in raw) / max(len(raw), 1)

    tok.enable_truncation(max_length=max_len)
    tok.enable_padding(length=max_len, pad_id=pad_id, pad_token=PAD)
    enc = tok.encode_batch(texts)
    ids = torch.tensor([e.ids for e in enc], dtype=torch.long)
    mask = torch.tensor([e.attention_mask for e in enc], dtype=torch.long)
    return ids, mask, truncated
