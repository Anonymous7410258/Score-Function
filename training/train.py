"""
Training Entry Point
====================
CLI-driven training script for Meta-TTA-TSM.

Orchestrates data generation, model instantiation, and meta-training.

Usage:
    python -m training.train --config configs/config.yaml --dataset gaussian
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np

from models.score_network import ScoreNetwork
from models.hypernetwork import HyperNetwork
from models.topology import TopologyExtractor
from training.meta_trainer import MetaTrainer
from data.datasets import DATASET_GENERATORS
from data.missingness import MissingnessTaskSampler
from utils.logger import ExperimentLogger
from utils.reproducibility import set_seed


def load_config(config_path: str, training_path: str = None) -> dict:
    """Load and merge configuration files.

    Args:
        config_path: Path to main config YAML.
        training_path: Optional path to training config YAML.

    Returns:
        Merged configuration dictionary.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if training_path and os.path.exists(training_path):
        with open(training_path, "r") as f:
            training_config = yaml.safe_load(f)
        config.update(training_config)

    return config


def resolve_device(device_str: str) -> torch.device:
    """Resolve device string to torch.device.

    Args:
        device_str: Device string ('auto', 'cuda', 'cpu').

    Returns:
        Resolved torch.device.
    """
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(
        description="Meta-TTA-TSM: Meta-Topological Test-Time Adaptive Score Matching"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to main configuration file.",
    )
    parser.add_argument(
        "--training-config",
        type=str,
        default="configs/training.yaml",
        help="Path to training configuration file.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset name (overrides config).",
    )
    parser.add_argument(
        "--missing-rate",
        type=float,
        default=None,
        help="Missingness rate (overrides config).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device: auto, cuda, cpu (overrides config).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Checkpoint directory (overrides config).",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from.",
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config, args.training_config)

    # Apply CLI overrides
    if args.dataset:
        config["data"]["dataset"] = args.dataset
    if args.missing_rate:
        config["missingness"]["train_missing_rate"] = args.missing_rate
    if args.device:
        config["device"] = args.device
    if args.seed:
        config["seed"] = args.seed

    # Setup
    seed = config.get("seed", 42)
    set_seed(seed)

    device = resolve_device(config.get("device", "auto"))

    checkpoint_dir = args.checkpoint_dir or config.get("output", {}).get(
        "checkpoint_dir", "checkpoints"
    )
    log_dir = config.get("output", {}).get("log_dir", "runs")

    # Initialize logger
    logger = ExperimentLogger(
        log_dir=log_dir,
        experiment_name=f"{config['data']['dataset']}_meta_training",
    )
    logger.info(f"Device: {device}")
    logger.info(f"Seed: {seed}")
    logger.info(f"Config: {config}")

    # --- Generate Data ---
    dataset_name = config["data"]["dataset"]
    data_dim = config["data"].get("data_dim", 50)
    n_samples = config["data"].get("num_samples", 5000)

    logger.info(f"Generating '{dataset_name}' dataset (dim={data_dim}, n={n_samples})...")

    gen_fn = DATASET_GENERATORS[dataset_name]
    gen_kwargs = {"n_samples": n_samples, "dim": data_dim, "seed": seed}

    result = gen_fn(**gen_kwargs)
    if isinstance(result, tuple):
        full_data = result[0]
    else:
        full_data = result

    # Train/val split
    train_ratio = config["data"].get("train_ratio", 0.8)
    split_idx = int(n_samples * train_ratio)
    train_data = full_data[:split_idx]
    val_data = full_data[split_idx:]

    logger.info(f"Train: {train_data.shape}, Val: {val_data.shape}")

    # --- Create Task Samplers ---
    miss_cfg = config.get("missingness", {})
    train_rate = miss_cfg.get("train_missing_rate", 0.4)
    missing_rates = [
        max(0.1, train_rate - 0.2),
        train_rate - 0.1,
        train_rate,
        train_rate + 0.1,
        min(0.9, train_rate + 0.2),
    ]

    train_task_sampler = MissingnessTaskSampler(
        data=train_data,
        missing_rates=missing_rates,
        samples_per_task=min(200, split_idx // 10),
        seed=seed,
    )

    val_task_sampler = MissingnessTaskSampler(
        data=val_data,
        missing_rates=missing_rates,
        samples_per_task=min(200, (n_samples - split_idx) // 5),
        seed=seed + 1000,
    )

    # --- Build Models ---
    # Update config with actual data dimensions
    config["model"]["score_network"]["input_dim"] = data_dim

    # Topology feature dimension
    topo_cfg = config["model"]["topology"]
    res = topo_cfg.get("persistence_image_resolution", [10, 10])
    max_hom = topo_cfg.get("max_homology_dim", 1)
    topo_feat_dim = res[0] * res[1] * (max_hom + 1)
    config["model"]["hypernetwork"]["topo_feature_dim"] = topo_feat_dim

    score_network = ScoreNetwork.from_config(config)
    topology_extractor = TopologyExtractor.from_config(config)
    hypernetwork = HyperNetwork.from_config(config, score_network)

    logger.info(f"Score Network params: {score_network.get_num_params():,}")
    logger.info(
        f"Hypernetwork params: "
        f"{sum(p.numel() for p in hypernetwork.parameters()):,}"
    )
    logger.info(f"Topology feature dim: {topo_feat_dim}")

    # --- Create Trainer ---
    trainer = MetaTrainer(
        score_network=score_network,
        hypernetwork=hypernetwork,
        topology_extractor=topology_extractor,
        config=config,
        device=device,
        logger=logger,
    )

    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # --- Train ---
    history = trainer.train(
        task_sampler=train_task_sampler,
        val_task_sampler=val_task_sampler,
        checkpoint_dir=checkpoint_dir,
    )

    logger.info(f"Training complete. Best meta loss: {trainer.best_meta_loss:.6f}")
    logger.info(f"Checkpoints saved to: {checkpoint_dir}")
    logger.close()


if __name__ == "__main__":
    main()
