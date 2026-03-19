"""Training loop for the hierarchical TabICL model.

Single-stage training — no curriculum or staged-sequence-length scheduling.

Speedups
--------
* torch.compile with shape bucketing (power-of-2 sizes, ~42 unique shapes
  instead of thousands — each new shape triggers a ~30 s recompilation)
* bfloat16 mixed precision via torch.autocast (no GradScaler; bf16 has the
  same exponent range as fp32 so overflow is not a concern)
* Fused AdamW kernel on CUDA
* Warmup → Constant → Cosine-Decay (WCD) schedule with lr_min floor
* EMA-based loss spike detection: skips steps where loss > threshold × EMA
* NaN / Inf loss guard
* OOM catch-and-skip (clears CUDA cache, continues training)
* Per-episode column permutation (anti-memorisation of column ordering)
* Per-episode label permutation for classification (anti-memorisation of
  quantile-based class ordering)
* Context-only y normalisation for regression (matches inference statistics)
* Gradient accumulation
"""

import math
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .model import TabICLModel
from .prior import generate_batch


__all__ = ['train', 'load_checkpoint']


# ---------------------------------------------------------------------------
# Shape buckets for torch.compile
# ---------------------------------------------------------------------------

# Power-of-2 (or nearby) buckets for (n_samples, n_features).
# Inputs are snapped DOWN to the largest fitting bucket before every step,
# so the compiled model sees only these shapes → minimal recompilations.
SAMPLE_BUCKETS  = [64, 128, 256, 512, 768, 1024, 1536, 2048]
FEATURE_BUCKETS = [4, 8, 16, 32, 48, 64, 96, 128]


def _snap_to_bucket(value, buckets, minimum):
    valid = [b for b in buckets if b <= value and b >= minimum]
    return max(valid) if valid else max(minimum, buckets[0])


# ---------------------------------------------------------------------------
# LR scheduler — Warmup → Constant → Cosine Decay
# ---------------------------------------------------------------------------

def _wcd_schedule(optimizer, warmup_steps, decay_start_step, total_steps, lr_min):
    """Linear warmup, optional constant plateau, then cosine decay to lr_min."""
    eff_decay_start = max(decay_start_step, warmup_steps)

    def lr_fn(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        if step < eff_decay_start:
            return 1.0
        progress = (step - eff_decay_start) / max(total_steps - eff_decay_start, 1)
        cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(lr_min / optimizer.defaults['lr'], cosine)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _unwrap(model):
    """Strip torch.compile (_orig_mod) wrapper to access raw TabICLModel."""
    return getattr(model, '_orig_mod', model)


def _save_checkpoint(model, optimizer, scheduler, step, path, bare_model=None):
    m = bare_model or _unwrap(model)
    torch.save({
        'step':            step,
        'model_state':     m.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'config': dict(
            max_features=m.max_features,
            embed_dim=m.embed_dim,
            max_classes=m.max_classes,
            nhead=m.icl.enc.layers[0].self_attn.num_heads,
            num_row_blocks=len(m.row_enc.enc.layers),
            num_icl_blocks=len(m.icl.enc.layers),
        ),
    }, path)


def load_checkpoint(path, model: TabICLModel | None = None,
                    device='cpu') -> tuple[TabICLModel, dict]:
    """Load a checkpoint and return ``(model, ckpt_dict)``.

    If *model* is None, a new :class:`TabICLModel` is created from the saved
    config.
    """
    from .model import create_model
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if model is None:
        model = create_model(**ckpt.get('config', {}))
    _unwrap(model).load_state_dict(ckpt['model_state'])
    return model.to(device), ckpt


# ---------------------------------------------------------------------------
# Batch preparation helpers
# ---------------------------------------------------------------------------

def _permute_columns(X_batch, rng):
    """Shuffle feature columns independently per episode.

    The SCM places root nodes in early columns and downstream nodes later.
    Without shuffling the model can learn "attend to later columns" as a
    shortcut rather than discovering relationships in-context.
    """
    B = X_batch.shape[0]
    F = X_batch.shape[2]
    for b in range(B):
        X_batch[b] = X_batch[b][:, rng.permutation(F)]
    return X_batch


def _permute_labels(y_batch, n_classes, rng):
    """Apply a random bijection to class labels per episode.

    Without this, class 0 always = lowest quantile and class N = highest.
    The model can learn "class ordering ≈ magnitude ordering" as a shortcut.
    """
    B = y_batch.shape[0]
    for b in range(B):
        perm       = rng.permutation(n_classes)
        y_int      = y_batch[b].astype(np.int64).clip(0, n_classes - 1)
        y_batch[b] = perm[y_int].astype(np.float32)
    return y_batch


def _normalise_reg_context(y_batch, n_ctx):
    """Re-normalise regression targets using context-only statistics.

    The data generator normalises with global (all-row) stats, but at
    inference we normalise with training-set-only stats.  Aligning these
    during training removes a systematic scale/shift mismatch that hurts R².
    """
    ctx_y    = y_batch[:, :n_ctx]
    ctx_mean = ctx_y.mean(axis=-1, keepdims=True)
    ctx_std  = ctx_y.std(axis=-1,  keepdims=True)
    ctx_std  = np.where(ctx_std < 1e-8, 1.0, ctx_std)
    return (y_batch - ctx_mean) / ctx_std


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def train(
    model: TabICLModel,
    *,
    # Optimisation
    n_steps:              int   = 100_000,
    batch_size:           int   = 32,
    lr:                   float = 1e-4,
    lr_min:               float = 0.0,
    weight_decay:         float = 1e-2,
    betas:                tuple = (0.9, 0.999),
    warmup_steps:         int   = 2_000,
    decay_start_step:     int   = 0,
    grad_clip:            float = 1.0,
    grad_accumulation:    int   = 1,
    # Compilation / precision
    compile:              bool  = False,
    compile_mode:         str   = 'default',
    mixed_precision:      bool  = True,
    # Infrastructure
    device:               str   = 'auto',
    save_path:            str   = 'hierarchical_tabicl.pt',
    log_interval:         int   = 100,
    save_interval:        int   = 5_000,
    resume:               bool  = True,
    # Data generation
    min_samples:          int   = 64,
    max_samples:          int   = 512,
    min_features:         int   = 1,
    max_features:         int   = 100,
    min_classes:          int   = 2,
    max_classes:          int   = 10,
    ctx_fraction_min:     float = 0.5,
    ctx_fraction_max:     float = 0.8,
    apply_filter:         bool  = False,
    # Robustness
    loss_spike_threshold: float = 10.0,
) -> TabICLModel:
    """Train *model* on the hierarchical LCS prior (single-stage).

    Parameters
    ----------
    model : TabICLModel
    n_steps : int
        Total gradient update steps.
    batch_size : int
        Synthetic datasets per step.
    lr : float
        Peak AdamW learning rate.
    lr_min : float
        Minimum LR at the end of cosine decay (absolute, not relative).
    weight_decay : float
        AdamW weight decay.
    betas : tuple
        AdamW beta coefficients.
    warmup_steps : int
        Linear LR warmup steps.
    decay_start_step : int
        Step at which cosine decay begins (0 = immediately after warmup).
    grad_clip : float
        Max gradient norm (0 = disabled).
    grad_accumulation : int
        Accumulate gradients over this many steps before an optimizer step.
    compile : bool
        Wrap the model with ``torch.compile``.  Adds a one-time compilation
        overhead (~1–3 min) but gives 20–40 % throughput improvement for
        large models on CUDA.  Inputs are snapped to shape buckets to
        minimise recompilations.
    compile_mode : str
        ``torch.compile`` mode (``'default'``, ``'reduce-overhead'``,
        ``'max-autotune'``).
    mixed_precision : bool
        Use bfloat16 ``torch.autocast`` on CUDA.  No GradScaler needed.
    device : str
        ``'auto'``, ``'cuda'``, or ``'cpu'``.
    save_path : str
        Checkpoint file path.
    log_interval : int
        Print training stats every N steps.
    save_interval : int
        Save checkpoint every N steps.
    resume : bool
        Resume from *save_path* if it exists.
    min/max_samples : int
        Range for randomly sampled dataset sizes.
    min/max_features : int
        Range for randomly sampled feature counts.
    min/max_classes : int
        Range for randomly sampled class counts.
    ctx_fraction_min / ctx_fraction_max : float
        Range for the context/query split ratio, sampled uniformly each step.
    apply_filter : bool
        Run ExtraTrees learnability filter on generated data (slower but
        rejects ~30 % of unlearnable datasets).
    loss_spike_threshold : float
        Skip a step if ``loss > loss_spike_threshold × loss_ema``.

    Returns
    -------
    TabICLModel
        Trained model (same object as *model*, moved to *device*).
    """
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = model.to(device)

    # ------------------------------------------------------------------
    # torch.compile
    # ------------------------------------------------------------------
    if compile:
        if not torch.cuda.is_available():
            warnings.warn('torch.compile requested but CUDA unavailable — skipping.', stacklevel=2)
        else:
            print(f"Compiling model (mode='{compile_mode}') …")
            t0 = time.time()
            model = torch.compile(model, mode=compile_mode)
            print(f"  Compilation took {time.time() - t0:.1f}s")

    bare_model = _unwrap(model)

    # ------------------------------------------------------------------
    # Optimizer and scheduler
    # ------------------------------------------------------------------
    _kw = {}
    try:                          # fused kernel available on PyTorch ≥ 2.0
        torch.optim.AdamW([], fused=True)
        if device == 'cuda':
            _kw['fused'] = True
    except TypeError:
        pass

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas, **_kw)
    scheduler = _wcd_schedule(optimizer, warmup_steps, decay_start_step,
                               n_steps * grad_accumulation, lr_min)

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------
    start_step = 0
    if resume and Path(save_path).exists():
        _, ckpt = load_checkpoint(save_path, model=bare_model, device=device)
        optimizer.load_state_dict(ckpt['optimizer_state'])
        scheduler.load_state_dict(ckpt['scheduler_state'])
        start_step = ckpt.get('step', 0)
        print(f"Resumed from step {start_step:,}")

    # ------------------------------------------------------------------
    # Mixed precision context
    # ------------------------------------------------------------------
    amp_ctx = (torch.autocast(device_type='cuda', dtype=torch.bfloat16)
               if mixed_precision and device == 'cuda'
               else torch.autocast(device_type='cpu', enabled=False))

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------
    rng     = np.random.default_rng(seed=0)
    model.train()

    run_loss    = 0.0
    run_acc     = 0.0
    run_skipped = 0
    loss_ema    = None
    micro_step  = 0   # counts individual forward passes for grad_accumulation

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    for step in range(start_step, n_steps):

        # --- Sample dataset dimensions (log-uniform in each range) ---
        log_ns = rng.uniform(np.log(min_samples), np.log(max_samples))
        log_nf = rng.uniform(np.log(min_features), np.log(max(min_features, max_features)))
        n_samp = int(np.exp(log_ns))
        n_feat = int(np.exp(log_nf))
        n_cls  = int(rng.integers(min_classes, max_classes + 1))
        n_cls  = min(n_cls, bare_model.max_classes)
        ctx_fr = rng.uniform(ctx_fraction_min, ctx_fraction_max)

        if compile:
            n_samp = _snap_to_bucket(n_samp, SAMPLE_BUCKETS,  min_samples)
            n_feat = _snap_to_bucket(n_feat, FEATURE_BUCKETS, min_features)

        # --- Generate data ---
        try:
            Xb, yb, n_cls_act = generate_batch(
                batch_size, n_samp, n_feat, 'cls',
                n_classes=n_cls, rng=rng, apply_filter=apply_filter,
            )
        except Exception as exc:
            warnings.warn(f'[step {step}] data gen failed: {exc}', stacklevel=2)
            run_skipped += 1
            continue

        n_cls_act = n_cls_act or n_cls

        # --- Column and label permutation (anti-memorisation) ---
        Xb = _permute_columns(Xb, rng)
        yb = _permute_labels(yb, n_cls_act, rng)

        # --- Context / query split ---
        n_ctx = max(n_cls_act, int(n_samp * ctx_fr))
        n_ctx = min(n_ctx, n_samp - 1)

        # Context-only y normalisation for regression alignment
        yb = _normalise_reg_context(yb, n_ctx)

        X_ctx = torch.tensor(Xb[:, :n_ctx],                  dtype=torch.float32, device=device)
        y_ctx = torch.tensor(yb[:, :n_ctx].astype(np.int64), device=device)
        X_qry = torch.tensor(Xb[:, n_ctx:],                  dtype=torch.float32, device=device)
        y_qry = torch.tensor(yb[:, n_ctx:].astype(np.int64), device=device)
        y_ctx = y_ctx.clamp(0, bare_model.max_classes - 1)
        y_qry = y_qry.clamp(0, bare_model.max_classes - 1)

        # --- Forward pass ---
        skip = False
        loss = None
        try:
            with amp_ctx:
                logits = model(X_ctx, y_ctx, X_qry)            # [B, n_qry, C]

            # Loss computed outside autocast for fp32 precision
            loss = F.cross_entropy(logits.flatten(0, 1), y_qry.flatten())
            loss_val = loss.item()

            if not math.isfinite(loss_val):
                warnings.warn(f'[step {step}] NaN/Inf loss — skipping', stacklevel=2)
                skip = True
            elif (loss_ema is not None
                  and loss_spike_threshold > 0
                  and loss_val > loss_spike_threshold * loss_ema):
                warnings.warn(
                    f'[step {step}] Loss spike {loss_val:.3f} > '
                    f'{loss_spike_threshold}×EMA({loss_ema:.3f}) — skipping',
                    stacklevel=2)
                skip = True

        except RuntimeError as exc:
            if 'out of memory' in str(exc).lower():
                warnings.warn(f'[step {step}] OOM — skipping', stacklevel=2)
                torch.cuda.empty_cache()
                skip = True
            else:
                raise
        except Exception as exc:
            warnings.warn(f'[step {step}] forward error: {exc}', stacklevel=2)
            skip = True

        if skip or loss is None:
            run_skipped += 1
            if grad_accumulation <= 1:
                optimizer.zero_grad(set_to_none=True)
            continue

        # Update EMA (non-skipped steps only)
        loss_ema = loss_val if loss_ema is None else 0.01 * loss_val + 0.99 * loss_ema

        # --- Backward ---
        scaled_loss = loss / grad_accumulation
        try:
            scaled_loss.backward()
        except RuntimeError as exc:
            if 'out of memory' in str(exc).lower():
                warnings.warn(f'[step {step}] OOM in backward — skipping', stacklevel=2)
                torch.cuda.empty_cache()
                optimizer.zero_grad(set_to_none=True)
                run_skipped += 1
                continue
            raise

        micro_step += 1
        run_loss += loss_val
        run_acc  += (logits.detach().flatten(0, 1).argmax(-1) == y_qry.flatten()).float().mean().item()

        # --- Optimizer step (every grad_accumulation micro-steps) ---
        if micro_step % grad_accumulation == 0:
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        # --- Logging ---
        if (step + 1) % log_interval == 0:
            n_logged = log_interval - run_skipped
            avg_loss = run_loss / max(n_logged, 1)
            avg_acc  = run_acc  / max(n_logged, 1)
            lr_now   = scheduler.get_last_lr()[0]
            print(f'[{step + 1:>7,d}/{n_steps:,}] '
                  f'loss={avg_loss:.4f}  acc={avg_acc:.3f}  '
                  f'lr={lr_now:.2e}  skipped={run_skipped}')
            run_loss = run_acc = run_skipped = 0

        # --- Checkpoint ---
        if (step + 1) % save_interval == 0:
            _save_checkpoint(model, optimizer, scheduler, step + 1,
                             save_path, bare_model=bare_model)
            print(f'  → checkpoint saved: {save_path}')

    # Final save
    _save_checkpoint(model, optimizer, scheduler, n_steps,
                     save_path, bare_model=bare_model)
    print(f'Training complete. Model saved to {save_path}')
    return model
