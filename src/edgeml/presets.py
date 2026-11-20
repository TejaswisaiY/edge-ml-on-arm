"""The two models we compare, and how each is trained.

``tiny``   - 2-layer transformer, own 4k-word vocabulary, random init, trained from scratch.
``minilm`` - 6-layer MiniLM (all-MiniLM-L6-v2), pretrained on large text corpora, fine-tuned here.

The question the comparison answers: for on-device intent understanding, does a model
~20x smaller trained only on this task get close enough to a pretrained one?
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

MAX_LEN = 32  # covers 100% of MiniLM-tokenised test sentences and 99.96% for the tiny vocab


@dataclass(frozen=True)
class TrainConfig:
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    grad_clip: float = 1.0
    label_smoothing: float = 0.0
    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Preset:
    name: str
    description: str
    train: TrainConfig
    # Tiny-model architecture (ignored for minilm, whose shape comes from its checkpoint).
    vocab_size: int = 4096
    hidden: int = 128
    layers: int = 2
    heads: int = 4
    intermediate: int = 512
    dropout: float = 0.1


PRESETS: dict[str, Preset] = {
    # Settings chosen by scripts/sweep_tiny.py on the validation split (results/sweeps/tiny_val.json).
    # The top two settings were within 0.1 points of each other (3 of 3,100 validation sentences),
    # i.e. within single-seed noise; the pre-declared rule (best validation accuracy) picked this one.
    "tiny": Preset(
        name="tiny",
        description="2-layer transformer, 4k WordPiece vocab, trained from scratch",
        train=TrainConfig(epochs=15, batch_size=64, lr=1e-3, label_smoothing=0.1),
        dropout=0.3,
    ),
    "minilm": Preset(
        name="minilm",
        description="all-MiniLM-L6-v2 (6 layers, 384 hidden), fine-tuned",
        train=TrainConfig(epochs=6, batch_size=32, lr=1e-4),
    ),
}
