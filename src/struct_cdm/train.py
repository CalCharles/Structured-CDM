"""Training loop for the hierarchical TabICL model.

Single-stage training — no curriculum or staged-sequence-length scheduling.
AdamW optimizer with linear warmup and cosine decay.
"""

import math
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .model import TabICLModel
from .prior import generate_batch


__all__ = ['train', 'load_checkpoint']


# ---------------------------------------------------------------------------
# LR scheduler
# ---------------------------------------------------------------------------

def _cosine_warmup(optimizer, warmup_steps: int, total_steps: int):
    def lr_fn(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        p = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * p)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _save_checkpoint(model, optimizer, scheduler, step, path):
    torch.save({
        'step':            step,
        'model_state':     model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'config': dict(
            max_features=model.max_features,
            embed_dim=model.embed_dim,
            max_classes=model.max_classes,
            nhead=model.icl.enc.layers[0].self_attn.num_heads,
            num_row_blocks=len(model.row_enc.enc.layers),
            num_icl_blocks=len(model.icl.enc.layers),
        ),
    }, path)


def load_checkpoint(path, model: TabICLModel | None = None,
                    device='cpu') -> tuple[TabICLModel, dict]:
    """Load a checkpoint and return (model, ckpt_dict).

    If *model* is None a new ``TabICLModel`` is created from the saved config.
    """
    from .model import create_model
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if model is None:
        model = create_model(**ckpt.get('config', {}))
    model.load_state_dict(ckpt['model_state'])
    return model.to(device), ckpt


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def train(
    model: TabICLModel,
    *,
    # Optimisation
    n_steps:       int   = 100_000,
    batch_size:    int   = 32,
    lr:            float = 1e-4,
    weight_decay:  float = 1e-2,
    warmup_steps:  int   = 2_000,
    grad_clip:     float = 1.0,
    # Infrastructure
    device:        str   = 'auto',
    save_path:     str   = 'hierarchical_tabicl.pt',
    log_interval:  int   = 100,
    save_interval: int   = 5_000,
    resume:        bool  = True,
    # Data generation
    min_samples:   int   = 64,
    max_samples:   int   = 512,
    min_features:  int   = 1,
    max_features:  int   = 100,
    min_classes:   int   = 2,
    max_classes:   int   = 10,
    ctx_fraction:  float = 0.7,
    apply_filter:  bool  = False,
) -> TabICLModel:
    """Train *model* on the hierarchical LCS prior (single-stage).

    Parameters
    ----------
    model : TabICLModel
    n_steps : int
        Total gradient update steps.
    batch_size : int
        Number of synthetic datasets per step.
    lr : float
        Peak learning rate for AdamW.
    weight_decay : float
        AdamW weight decay.
    warmup_steps : int
        Number of linear warmup steps before cosine decay begins.
    grad_clip : float
        Max gradient norm (0 = disabled).
    device : str
        ``'auto'``, ``'cuda'``, or ``'cpu'``.
    save_path : str
        Checkpoint file path.
    log_interval : int
        Print training stats every N steps.
    save_interval : int
        Save checkpoint every N steps.
    resume : bool
        If True and *save_path* exists, resume from that checkpoint.
    min_samples / max_samples : int
        Range for randomly sampled dataset sizes.
    min_features / max_features : int
        Range for randomly sampled feature counts.
    min_classes / max_classes : int
        Range for randomly sampled class counts.
    ctx_fraction : float
        Fraction of samples used as context (train split); rest are queries.
    apply_filter : bool
        Run ExtraTrees learnability filter on generated data. Improves data
        quality at the cost of ~2–3× slower data generation.

    Returns
    -------
    TabICLModel
        The trained model (same object as *model*, moved to *device*).
    """
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = model.to(device)

    # Fused AdamW on CUDA when available
    _kw = {}
    try:
        torch.optim.AdamW([], fused=True)
        if device == 'cuda':
            _kw['fused'] = True
    except TypeError:
        pass
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                   weight_decay=weight_decay, **_kw)
    scheduler = _cosine_warmup(optimizer, warmup_steps, n_steps)

    start_step = 0
    if resume and Path(save_path).exists():
        _, ckpt = load_checkpoint(save_path, model=model, device=device)
        optimizer.load_state_dict(ckpt['optimizer_state'])
        scheduler.load_state_dict(ckpt['scheduler_state'])
        start_step = ckpt.get('step', 0)
        print(f"Resumed from step {start_step:,}")

    rng      = np.random.default_rng(seed=0)
    model.train()
    run_loss = run_acc = 0.0

    for step in range(start_step, n_steps):
        # Sample dataset dimensions
        n_samp = int(rng.choice([min_samples,
                                  (min_samples + max_samples) // 2,
                                  max_samples]))
        n_feat = int(rng.integers(min_features, max_features + 1))
        n_cls  = int(rng.integers(min_classes, max_classes + 1))
        n_cls  = min(n_cls, model.max_classes)

        try:
            Xb, yb, n_cls_act = generate_batch(
                batch_size, n_samp, n_feat, 'cls',
                n_classes=n_cls, rng=rng, apply_filter=apply_filter,
            )
        except Exception as exc:
            warnings.warn(f"[step {step}] data gen failed: {exc}", stacklevel=2)
            continue

        n_cls_act = n_cls_act or n_cls
        n_ctx = max(n_cls_act, int(n_samp * ctx_fraction))
        n_ctx = min(n_ctx, n_samp - 1)

        X_ctx = torch.tensor(Xb[:, :n_ctx],                  dtype=torch.float32, device=device)
        y_ctx = torch.tensor(yb[:, :n_ctx].astype(np.int64), device=device)
        X_qry = torch.tensor(Xb[:, n_ctx:],                  dtype=torch.float32, device=device)
        y_qry = torch.tensor(yb[:, n_ctx:].astype(np.int64), device=device)
        y_ctx = y_ctx.clamp(0, model.max_classes - 1)
        y_qry = y_qry.clamp(0, model.max_classes - 1)

        try:
            logits = model(X_ctx, y_ctx, X_qry)              # [B, n_qry, max_classes]
            loss   = F.cross_entropy(logits.flatten(0, 1), y_qry.flatten())
        except Exception as exc:
            warnings.warn(f"[step {step}] forward failed: {exc}", stacklevel=2)
            continue

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        run_loss += loss.item()
        run_acc  += (logits.flatten(0, 1).argmax(-1) == y_qry.flatten()).float().mean().item()

        if (step + 1) % log_interval == 0:
            print(f"[{step + 1:>7,d}/{n_steps:,}] "
                  f"loss={run_loss / log_interval:.4f}  "
                  f"acc={run_acc / log_interval:.3f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")
            run_loss = run_acc = 0.0

        if (step + 1) % save_interval == 0:
            _save_checkpoint(model, optimizer, scheduler, step + 1, save_path)
            print(f"  → checkpoint saved: {save_path}")

    _save_checkpoint(model, optimizer, scheduler, n_steps, save_path)
    print(f"Training complete. Model saved to {save_path}")
    return model
