"""CLI entry point for training the hierarchical TabICL model.

Usage
-----
python scripts/run_train.py --steps 100000 --device cuda --save model.pt
python scripts/run_train.py --steps 100000 --embed-dim 256 --icl-blocks 12 --compile
"""

import argparse
import sys
from pathlib import Path

# Allow running directly without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from struct_cdm.model import create_model
from struct_cdm.train import train


def main():
    p = argparse.ArgumentParser(description='Train hierarchical TabICL')

    # Model
    g = p.add_argument_group('model')
    g.add_argument('--max-features', type=int,   default=100)
    g.add_argument('--embed-dim',    type=int,   default=128)
    g.add_argument('--nhead',        type=int,   default=8)
    g.add_argument('--row-blocks',   type=int,   default=3)
    g.add_argument('--icl-blocks',   type=int,   default=8)
    g.add_argument('--max-classes',  type=int,   default=10)
    g.add_argument('--dropout',      type=float, default=0.0)

    # Optimisation
    g = p.add_argument_group('optimisation')
    g.add_argument('--steps',             type=int,   default=100_000)
    g.add_argument('--batch-size',        type=int,   default=32)
    g.add_argument('--lr',                type=float, default=1e-4)
    g.add_argument('--lr-min',            type=float, default=0.0,
                   help='Minimum LR at end of cosine decay')
    g.add_argument('--weight-decay',      type=float, default=1e-2)
    g.add_argument('--warmup',            type=int,   default=2_000)
    g.add_argument('--decay-start',       type=int,   default=0,
                   help='Step at which cosine decay begins (0 = right after warmup)')
    g.add_argument('--grad-clip',         type=float, default=1.0)
    g.add_argument('--grad-accumulation', type=int,   default=1,
                   help='Accumulate gradients over N steps before optimizer step')

    # Infrastructure
    g = p.add_argument_group('infrastructure')
    g.add_argument('--device',             type=str,  default='auto')
    g.add_argument('--save',               type=str,  default='hierarchical_tabicl.pt')
    g.add_argument('--log-interval',       type=int,  default=100)
    g.add_argument('--save-interval',      type=int,  default=5_000)
    g.add_argument('--no-resume',          action='store_true')
    g.add_argument('--compile',            action='store_true',
                   help='Wrap model with torch.compile (~1-3 min overhead, 20-40%% faster)')
    g.add_argument('--compile-mode',       type=str,  default='default',
                   choices=['default', 'reduce-overhead', 'max-autotune'])
    g.add_argument('--no-mixed-precision', action='store_true',
                   help='Disable bfloat16 autocast (use fp32 throughout)')

    # Data
    g = p.add_argument_group('data')
    g.add_argument('--min-samples',           type=int,   default=64)
    g.add_argument('--max-samples',           type=int,   default=512)
    g.add_argument('--min-features',          type=int,   default=1)
    g.add_argument('--min-classes',           type=int,   default=2)
    g.add_argument('--ctx-fraction-min',      type=float, default=0.5)
    g.add_argument('--ctx-fraction-max',      type=float, default=0.8)
    g.add_argument('--loss-spike-threshold',  type=float, default=10.0,
                   help='Skip steps where loss > threshold * EMA (0 = disabled)')
    g.add_argument('--filter',                action='store_true',
                   help='Apply ExtraTrees learnability filter (slower, cleaner data)')

    args = p.parse_args()

    model = create_model(
        max_features=args.max_features,
        embed_dim=args.embed_dim,
        nhead=args.nhead,
        num_row_blocks=args.row_blocks,
        num_icl_blocks=args.icl_blocks,
        max_classes=args.max_classes,
        dropout=args.dropout,
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    train(
        model,
        n_steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        lr_min=args.lr_min,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup,
        decay_start_step=args.decay_start,
        grad_clip=args.grad_clip,
        grad_accumulation=args.grad_accumulation,
        compile=args.compile,
        compile_mode=args.compile_mode,
        mixed_precision=not args.no_mixed_precision,
        device=args.device,
        save_path=args.save,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume=not args.no_resume,
        min_samples=args.min_samples,
        max_samples=args.max_samples,
        min_features=args.min_features,
        max_features=args.max_features,
        min_classes=args.min_classes,
        ctx_fraction_min=args.ctx_fraction_min,
        ctx_fraction_max=args.ctx_fraction_max,
        loss_spike_threshold=args.loss_spike_threshold,
        apply_filter=args.filter,
    )


if __name__ == '__main__':
    main()
