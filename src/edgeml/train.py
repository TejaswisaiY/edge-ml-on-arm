"""Training loop, written out in plain PyTorch (no Trainer abstraction).

Each step: forward pass -> cross-entropy loss -> backward pass (gradients) ->
clip gradients -> AdamW update -> learning-rate schedule step.

Outputs, in ``artifacts/runs/<preset>/``:
    model.pt          best weights (chosen on the validation split, never on test)
    tokenizer.json    the exact tokenizer the model was trained with
    config.json       architecture + training settings
    metrics.json      validation and test metrics of the best checkpoint
    train_log.jsonl   one line per epoch
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import paths
from .data import load_clinc150
from .metrics import intent_metrics, predict
from .model import EncoderConfig, IntentClassifier, count_parameters, init_weights
from .presets import MAX_LEN, PRESETS, Preset, TrainConfig
from .pretrained import download_minilm, load_minilm_classifier
from .tokenization import encode, load_tokenizer, train_wordpiece


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def linear_warmup_decay(step: int, total: int, warmup: int) -> float:
    """LR multiplier: ramps 0 -> 1 over ``warmup`` steps, then decays linearly to 0."""
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    return max(0.0, (total - step) / max(total - warmup, 1))


def param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """No weight decay on biases, LayerNorm and embedding tables (standard for transformers)."""
    decay, no_decay = [], []
    for module in model.modules():
        for name, p in module.named_parameters(recurse=False):
            if isinstance(module, (nn.LayerNorm, nn.Embedding)) or name == "bias":
                no_decay.append(p)
            else:
                decay.append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def build(preset: Preset, num_labels: int, train_texts: list[str]):
    """Return ``(model, tokenizer)`` for a preset."""
    if preset.name == "minilm":
        root = download_minilm()
        return load_minilm_classifier(num_labels, MAX_LEN, root), load_tokenizer(root / "tokenizer.json")
    tok = train_wordpiece(train_texts, preset.vocab_size)
    cfg = EncoderConfig(
        vocab_size=tok.get_vocab_size(),
        hidden=preset.hidden,
        layers=preset.layers,
        heads=preset.heads,
        intermediate=preset.intermediate,
        max_positions=MAX_LEN,
        num_labels=num_labels,
        dropout=preset.dropout,
    )
    model = IntentClassifier(cfg)
    init_weights(model)
    return model, tok


def train(preset_name: str, out_dir: Path | None = None, threads: int | None = None, max_steps: int | None = None,
          overrides: dict | None = None, evaluate_test: bool = True) -> dict:
    """Train one preset. ``evaluate_test=False`` is used by hyper-parameter sweeps, which must
    choose settings on the validation split only and never look at test."""
    overrides = dict(overrides or {})
    preset = PRESETS[preset_name]
    if "dropout" in overrides:
        preset = dataclasses.replace(preset, dropout=overrides.pop("dropout"))
    tc = TrainConfig(**{**preset.train.to_dict(), **overrides})
    out_dir = Path(out_dir or paths.run_dir(preset_name))
    out_dir.mkdir(parents=True, exist_ok=True)
    if threads:
        torch.set_num_threads(threads)
    seed_everything(tc.seed)

    data = load_clinc150()
    model, tok = build(preset, data.num_labels, data.train.texts)
    tok.save(str(out_dir / "tokenizer.json"))

    tr_ids, tr_mask, _ = encode(tok, data.train.texts, MAX_LEN)
    va_ids, va_mask, _ = encode(tok, data.val.texts, MAX_LEN)
    te_ids, te_mask, te_trunc = encode(tok, data.test.texts, MAX_LEN)
    tr_y = torch.tensor(data.train.labels)
    va_y, te_y = torch.tensor(data.val.labels), torch.tensor(data.test.labels)

    steps_per_epoch = math.ceil(len(tr_y) / tc.batch_size)
    total_steps = steps_per_epoch * tc.epochs if max_steps is None else min(max_steps, steps_per_epoch * tc.epochs)
    warmup = int(tc.warmup_ratio * total_steps)
    opt = torch.optim.AdamW(param_groups(model, tc.weight_decay), lr=tc.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: linear_warmup_decay(s, total_steps, warmup))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=tc.label_smoothing)
    gen = torch.Generator().manual_seed(tc.seed)

    config = {"preset": preset_name, "description": preset.description, "max_len": MAX_LEN,
              "model": model.cfg.to_dict(), "train": tc.to_dict(), "parameters": count_parameters(model),
              "test_truncated_fraction": te_trunc, "torch": torch.__version__, "threads": torch.get_num_threads()}
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(json.dumps({k: config[k] for k in ("preset", "parameters", "train")}), flush=True)

    log_path = out_dir / "train_log.jsonl"
    log_path.write_text("")
    best_val, best_epoch, step = -1.0, -1, 0
    t_start = time.perf_counter()
    for epoch in range(1, tc.epochs + 1):
        model.train()
        order = torch.randperm(len(tr_y), generator=gen)
        running, n_seen, t_epoch = 0.0, 0, time.perf_counter()
        for i in range(0, len(order), tc.batch_size):
            if step >= total_steps:
                break
            idx = order[i : i + tc.batch_size]
            logits = model(tr_ids[idx], tr_mask[idx])
            loss = loss_fn(logits, tr_y[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
            opt.step()
            sched.step()
            step += 1
            running += loss.item() * len(idx)
            n_seen += len(idx)

        val = intent_metrics(predict(model, va_ids, va_mask), va_y, data.oos_id)
        entry = {"epoch": epoch, "step": step, "train_loss": running / max(n_seen, 1), "lr": sched.get_last_lr()[0],
                 "epoch_seconds": round(time.perf_counter() - t_epoch, 2), **{f"val_{k}": v for k, v in val.items()}}
        with open(log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        print(json.dumps(entry), flush=True)

        # Model selection uses the validation split only. Test is touched once, at the end.
        if val["overall_accuracy"] > best_val:
            best_val, best_epoch = val["overall_accuracy"], epoch
            torch.save(model.state_dict(), out_dir / "model.pt")
        if step >= total_steps:
            break

    model.load_state_dict(torch.load(out_dir / "model.pt", weights_only=True))
    val = intent_metrics(predict(model, va_ids, va_mask), va_y, data.oos_id)
    test = intent_metrics(predict(model, te_ids, te_mask), te_y, data.oos_id) if evaluate_test else None
    metrics = {"preset": preset_name, "best_epoch": best_epoch, "steps": step,
               "train_seconds": round(time.perf_counter() - t_start, 1), "val": val, "test": test}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics), flush=True)
    return metrics


def load_trained(preset_name: str, run_dir: Path | None = None):
    """Rebuild a trained model + tokenizer from a run directory."""
    run_dir = Path(run_dir or paths.run_dir(preset_name))
    cfg = json.loads((run_dir / "config.json").read_text())
    model = IntentClassifier(EncoderConfig(**cfg["model"]))
    model.load_state_dict(torch.load(run_dir / "model.pt", weights_only=True))
    return model.eval(), load_tokenizer(run_dir / "tokenizer.json"), cfg


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("preset", choices=sorted(PRESETS))
    ap.add_argument("--out", type=Path)
    ap.add_argument("--threads", type=int)
    ap.add_argument("--max-steps", type=int, help="stop early (smoke tests)")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--dropout", type=float, help="tiny model only")
    ap.add_argument("--label-smoothing", type=float)
    a = ap.parse_args(argv)
    overrides = {k: v for k, v in {"epochs": a.epochs, "lr": a.lr, "seed": a.seed, "dropout": a.dropout,
                                   "label_smoothing": a.label_smoothing}.items() if v is not None}
    train(a.preset, a.out, a.threads, a.max_steps, overrides)


if __name__ == "__main__":
    main()
