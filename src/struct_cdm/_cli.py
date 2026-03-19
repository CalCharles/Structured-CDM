"""Package-level CLI entry point (struct-sdfm-train)."""

# Re-export the main() function from the top-level train script so it can be
# registered as a console_scripts entry point without duplicating argument
# parsing logic.

import sys
import os

# Allow running from the repo root as well as via the installed entry point
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


def main():
    # Import here to avoid circular import at package load time
    import importlib.util
    from pathlib import Path

    train_script = Path(__file__).resolve().parent.parent.parent / 'train.py'
    if train_script.exists():
        spec   = importlib.util.spec_from_file_location('_train_cli', train_script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.main()
    else:
        # Fallback: inline minimal arg-parse
        from .model import create_model
        from .train import train
        import argparse

        p = argparse.ArgumentParser(description='Train hierarchical TabICL')
        p.add_argument('--steps',       type=int,   default=100_000)
        p.add_argument('--batch-size',  type=int,   default=32)
        p.add_argument('--embed-dim',   type=int,   default=128)
        p.add_argument('--icl-blocks',  type=int,   default=8)
        p.add_argument('--max-classes', type=int,   default=10)
        p.add_argument('--lr',          type=float, default=1e-4)
        p.add_argument('--device',      type=str,   default='auto')
        p.add_argument('--save',        type=str,   default='hierarchical_tabicl.pt')
        p.add_argument('--no-resume',   action='store_true')
        args = p.parse_args()

        model = create_model(embed_dim=args.embed_dim,
                              num_icl_blocks=args.icl_blocks,
                              max_classes=args.max_classes)
        train(model, n_steps=args.steps, batch_size=args.batch_size,
              lr=args.lr, device=args.device, save_path=args.save,
              resume=not args.no_resume)
