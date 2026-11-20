"""CLINC150 intent dataset (Larson et al., 2019), licensed CC BY 3.0.

150 everyday intents (reminders, calendar, timers, to-do lists, translation, banking...)
plus an "out-of-scope" (oos) class for requests the assistant should not try to handle.
That second part matters for an on-device companion: it has to know when it does *not*
understand, instead of confidently doing the wrong thing.

We use the "full" variant: 100 training examples per intent, plus the official oos splits.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .download import fetch
from .paths import DATA_DIR

CLINC_COMMIT = "828f8093932c8fe6ca7936c3d2e52903b1c523de"
CLINC_URL = f"https://raw.githubusercontent.com/clinc/oos-eval/{CLINC_COMMIT}/data/data_full.json"
CLINC_SHA256 = "36923c3705a59e08fe9c3883d8bc2dd966ef93e22cb78ac41171782a698d56e0"
OOS = "oos"


@dataclass(frozen=True)
class Split:
    texts: list[str]
    labels: list[int]

    def __len__(self) -> int:
        return len(self.texts)


@dataclass(frozen=True)
class Clinc150:
    train: Split
    val: Split
    test: Split
    label_names: list[str]

    @property
    def num_labels(self) -> int:
        return len(self.label_names)

    @property
    def oos_id(self) -> int:
        return self.label_names.index(OOS)


def load_clinc150(cache_dir: Path = DATA_DIR) -> Clinc150:
    path = fetch(CLINC_URL, Path(cache_dir) / "clinc150_data_full.json", CLINC_SHA256)
    raw = json.loads(path.read_text(encoding="utf-8"))

    # Sorted in-scope intents, then "oos" last, so label ids are stable across runs.
    in_scope = sorted({intent for _, intent in raw["train"]})
    label_names = [*in_scope, OOS]
    index = {name: i for i, name in enumerate(label_names)}

    def split(name: str) -> Split:
        rows = raw[name] + raw[f"oos_{name}"]
        return Split(texts=[t for t, _ in rows], labels=[index[l] for _, l in rows])

    return Clinc150(train=split("train"), val=split("val"), test=split("test"), label_names=label_names)
