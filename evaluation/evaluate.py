"""
Evaluation Script
=================
Runs evaluation across datasets and missingness regimes, producing
result tables matching the experimental protocol in Section 5 of the paper.

Usage:
    python -m evaluation.evaluate \
        --config configs/config.yaml \
        --checkpoint checkpoints/best_model.pt \
        --dataset gaussian \
        --missing-rates 0.2 0.4 0.6 0.8
"""

import os
import argparse
import yaml
import torch
import numpy as np
from typing import List, Dict

from models.score_network import ScoreNetwork
from models.hypernetwork import HyperNetwork
from models.topology import TopologyExtractor
from inference.tta import TestTimeAdapter
from inference.predict import load_checkpoint
from data.datasets import DATASET_GENERATORS
from data.missingness import generate_mcar_mask
from evaluation.metrics import (
    fisher_divergence,
    fisher_divergence_gaussian,
    mmd,
    structure_recovery_metrics,
)
from utils.reproducibility import set_seed


def evaluate_gaussian(
    score_network: ScoreNetwork,
    hypernetwork: HyperNetwork,
    topology_extractor: TopologyExtractor,
    config: dict,
    device: torch.device,
    missing_rates: List[float],
    truncated: bool = False,
    n_seeds: int = 5,
    n_samples: int = 5000,
    dim: int = 50,
) -> Dict:
    """Evaluate on Gaussian datasets across missingness rates.

    Reference: Table 3 in the paper.

    Args:
        score_network: Score network.
        hypernetwork: Trained hypernetwork.
        topology_extractor: Topology extractor.
        config: Configuration.
        device: Torch device.
        missing_rates: List of missingness rates to evaluate.
        truncated: Use truncated Gaussian.
        n_seeds: Number of random seeds.
        n_samples: Samples per experiment.
        dim: Data dimensionality.

    Returns:
        Dictionary of results per missingness rate.
    """
    results = {}

    for rate in missing_rates:
        rate_results = []

        for seed in range(n_seeds):
            set_seed(seed)

            # Generate data with known parameters
            rng = np.random.RandomState(seed)
            A = rng.randn(dim, dim) * 0.5
            cov = A @ A.T + np.eye(dim) * 0.1
            mean = rng.randn(dim) * 0.1
            precision = np.linalg.inv(cov)

            data = rng.multivariate_normal(mean, cov, size=n_samples)
            if truncated:
                data = np.abs(data)

            data = data.astype(np.float32)

            # Generate masks
            masks = generate_mcar_mask(data.shape, rate, seed=seed)

            # Split train/test
            split = int(n_samples * 0.8)
            train_data = data[:split] * masks[:split]
            test_data = data[split:]
            test_masks = masks[split:]

            # Initialize adapter
            adapter = TestTimeAdapter(
                score_network=score_network,
                hypernetwork=hypernetwork,
                topology_extractor=topology_extractor,
                config=config,
                device=device,
            )
            adapter.set_training_topology(train_data)

            # Predict
            result = adapter.predict_with_tta(
                test_data_np=test_data * test_masks,
                test_masks_np=test_masks,
                verbose=False,
            )

            # Compute Fisher divergence against oracle
            fd = fisher_divergence_gaussian(
                result["scores"],
                test_data,
                mean,
                precision,
                test_masks,
            )
            rate_results.append(fd)

        results[rate] = {
            "mean": np.mean(rate_results),
            "std": np.std(rate_results),
        }

    return results


def print_results_table(
    results: Dict,
    title: str = "Results",
):
    """Print formatted results table.

    Args:
        results: Dictionary mapping metrics to values.
        title: Table title.
    """
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"{'Missing Rate':<15} {'Fisher Divergence':<25}")
    print(f"{'-'*40}")

    for rate in sorted(results.keys()):
        r = results[rate]
        print(f"{rate*100:>5.0f}%{'':>9} {r['mean']:.6f} ± {r['std']:.6f}")

    print(f"{'='*60}\n")


def main():
    """CLI entry point for evaluation."""
    parser = argparse.ArgumentParser(
        description="Meta-TTA-TSM: Evaluation"
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--dataset",
        type=str,
        default="gaussian",
        choices=list(DATASET_GENERATORS.keys()),
    )
    parser.add_argument(
        "--missing-rates",
        type=float,
        nargs="+",
        default=[0.2, 0.4, 0.6, 0.8],
    )
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--truncated", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    training_cfg = os.path.join(os.path.dirname(args.config), "training.yaml")
    if os.path.exists(training_cfg):
        with open(training_cfg, "r") as f:
            config.update(yaml.safe_load(f))

    set_seed(args.seed)
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )

    print(f"Device: {device}")
    print(f"Dataset: {args.dataset}")
    print(f"Missing rates: {args.missing_rates}")

    # Load model
    score_network, hypernetwork, topology_extractor = load_checkpoint(
        args.checkpoint, config, device
    )

    # Run evaluation
    if args.dataset in ("gaussian", "gaussian_truncated"):
        results = evaluate_gaussian(
            score_network=score_network,
            hypernetwork=hypernetwork,
            topology_extractor=topology_extractor,
            config=config,
            device=device,
            missing_rates=args.missing_rates,
            truncated=args.truncated or args.dataset == "gaussian_truncated",
            n_seeds=args.n_seeds,
            dim=config.get("data", {}).get("data_dim", 50),
        )

        variant = "Truncated" if args.truncated else "Non-truncated"
        print_results_table(results, f"Fisher Divergence — {variant} Gaussian")
    else:
        print(
            f"Evaluation for '{args.dataset}' follows the same protocol. "
            f"See evaluation/metrics.py for available metrics."
        )

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    result_path = os.path.join(
        args.output_dir, f"{args.dataset}_results.npy"
    )
    np.save(result_path, results)
    print(f"Results saved to: {result_path}")


if __name__ == "__main__":
    main()
