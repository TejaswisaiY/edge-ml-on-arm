"""One BERT-style encoder, used at two sizes.

The *same* class is used for the tiny model (trained from scratch) and for MiniLM
(pretrained weights loaded in). That keeps the comparison fair: any difference in
speed or accuracy comes from size and training, not from different code.

Written out by hand, layer by layer, so every operation that ends up on the Arm CPU
is visible here: embeddings -> N x (self-attention, feed-forward) -> mean pooling -> classifier.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class EncoderConfig:
    vocab_size: int
    hidden: int
    layers: int
    heads: int
    intermediate: int
    max_positions: int
    num_labels: int
    dropout: float = 0.1
    ln_eps: float = 1e-12

    def __post_init__(self) -> None:
        if self.hidden % self.heads:
            raise ValueError(f"hidden ({self.hidden}) must be divisible by heads ({self.heads})")

    def to_dict(self) -> dict:
        return asdict(self)


class Embeddings(nn.Module):
    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()
        self.word = nn.Embedding(cfg.vocab_size, cfg.hidden)
        self.position = nn.Embedding(cfg.max_positions, cfg.hidden)
        # Single-sentence inputs only, so only segment 0 is ever used. Kept as a
        # 2-row table so pretrained BERT/MiniLM weights load unchanged.
        self.token_type = nn.Embedding(2, cfg.hidden)
        self.norm = nn.LayerNorm(cfg.hidden, eps=cfg.ln_eps)
        self.dropout = nn.Dropout(cfg.dropout)
        self.register_buffer("position_ids", torch.arange(cfg.max_positions).unsqueeze(0), persistent=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_len = input_ids.shape[1]
        x = self.word(input_ids) + self.position(self.position_ids[:, :seq_len]) + self.token_type.weight[0]
        return self.dropout(self.norm(x))


class SelfAttention(nn.Module):
    """Multi-head self-attention, post-LayerNorm (as in BERT)."""

    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()
        self.heads = cfg.heads
        self.head_dim = cfg.hidden // cfg.heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.query = nn.Linear(cfg.hidden, cfg.hidden)
        self.key = nn.Linear(cfg.hidden, cfg.hidden)
        self.value = nn.Linear(cfg.hidden, cfg.hidden)
        self.out = nn.Linear(cfg.hidden, cfg.hidden)
        self.norm = nn.LayerNorm(cfg.hidden, eps=cfg.ln_eps)
        self.dropout = nn.Dropout(cfg.dropout)

    def _split(self, t: torch.Tensor) -> torch.Tensor:
        b, l, _ = t.shape
        return t.reshape(b, l, self.heads, self.head_dim).transpose(1, 2)  # [B, H, L, D]

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor) -> torch.Tensor:
        b, l, h = x.shape
        q, k, v = self._split(self.query(x)), self._split(self.key(x)), self._split(self.value(x))
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale + attn_bias  # [B, H, L, L]
        probs = self.dropout(torch.softmax(scores, dim=-1))
        ctx = torch.matmul(probs, v).transpose(1, 2).reshape(b, l, h)
        return self.norm(x + self.dropout(self.out(ctx)))


class FeedForward(nn.Module):
    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()
        self.up = nn.Linear(cfg.hidden, cfg.intermediate)
        self.act = nn.GELU()
        self.down = nn.Linear(cfg.intermediate, cfg.hidden)
        self.norm = nn.LayerNorm(cfg.hidden, eps=cfg.ln_eps)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.dropout(self.down(self.act(self.up(x)))))


class EncoderLayer(nn.Module):
    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()
        self.attn = SelfAttention(cfg)
        self.ffn = FeedForward(cfg)

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor) -> torch.Tensor:
        return self.ffn(self.attn(x, attn_bias))


class Encoder(nn.Module):
    """Token ids -> contextual vectors ``[B, L, hidden]``."""

    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()
        self.embeddings = Embeddings(cfg)
        self.layers = nn.ModuleList(EncoderLayer(cfg) for _ in range(cfg.layers))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = self.embeddings(input_ids)
        # Padding positions get a large negative score so softmax gives them ~0 weight.
        mask = attention_mask.to(x.dtype)
        attn_bias = (1.0 - mask)[:, None, None, :] * -10000.0
        for layer in self.layers:
            x = layer(x, attn_bias)
        return x


class IntentClassifier(nn.Module):
    """Encoder + masked mean pooling + linear head. Output: logits ``[B, num_labels]``."""

    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg)
        self.dropout = nn.Dropout(cfg.dropout)
        self.classifier = nn.Linear(cfg.hidden, cfg.num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = self.encoder(input_ids, attention_mask)
        mask = attention_mask.to(x.dtype).unsqueeze(-1)
        pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return self.classifier(self.dropout(pooled))


def init_weights(model: nn.Module, std: float = 0.02) -> None:
    """BERT-style initialisation, used when training from scratch."""
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=std)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=std)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Parameter counts split into embedding tables vs everything else (the compute)."""
    emb = sum(p.numel() for m in model.modules() if isinstance(m, nn.Embedding) for p in m.parameters())
    total = sum(p.numel() for p in model.parameters())
    return {"total": total, "embeddings": emb, "non_embedding": total - emb}
