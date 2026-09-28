"""Train an SAE from a config file.

Example:
    python train.py --config configs/main/tree_sae_4layer_l0_32.yaml --device cuda:0
"""

import argparse

from tree_sae.training.config import TrainConfig
from tree_sae.training.trainer import train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="YAML file in configs/")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_dir", default=None, help="Overrides the config's output_dir")
    parser.add_argument("--max_steps", type=int, default=None, help="Stop early (for debugging)")
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--no_wandb", action="store_true", help="Disable Weights & Biases logging")
    args = parser.parse_args()

    overrides = {k: v for k, v in (("output_dir", args.output_dir), ("num_workers", args.num_workers)) if v is not None}
    cfg = TrainConfig.from_yaml(args.config, **overrides)
    train(cfg, device=args.device, max_steps=args.max_steps, use_wandb=not args.no_wandb)


if __name__ == "__main__":
    main()
