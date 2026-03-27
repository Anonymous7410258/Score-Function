"""
Prediction Module
=================
CLI-driven score prediction with optional test-time adaptation.

Usage:
    python -m inference.predict \
        --config configs/config.yaml \
        --checkpoint checkpoints/best_model.pt \
        --input data/processed/test_data.npy \
        --enable-tta \
        --output results/scores.npy
"""

import os
import argparse
import yaml
import torch
import numpy as np

from models.score_network import ScoreNetwork
from models.hypernetwork import HyperNetwork
from models.topology import TopologyExtractor
from inference.tta import TestTimeAdapter
from utils.reproducibility import set_seed


def load_checkpoint(
    checkpoint_path: str,
    config: dict,
    device: torch.device,
):
    """Load model from checkpoint.

    Args:
        checkpoint_path: Path to .pt checkpoint file.
        config: Configuration dictionary.
        device: Torch device.

    Returns:
        Tuple of (score_network, hypernetwork, topology_extractor).
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Rebuild models from config
    score_network = ScoreNetwork.from_config(config).to(device)
    topology_extractor = TopologyExtractor.from_config(config)

    # Compute topo feature dim
    topo_cfg = config["model"]["topology"]
    res = topo_cfg.get("persistence_image_resolution", [10, 10])
    max_hom = topo_cfg.get("max_homology_dim", 1)
    config["model"]["hypernetwork"]["topo_feature_dim"] = (
        res[0] * res[1] * (max_hom + 1)
    )

    hypernetwork = HyperNetwork.from_config(config, score_network).to(device)
    hypernetwork.load_state_dict(checkpoint["hypernetwork_state_dict"])

    return score_network, hypernetwork, topology_extractor


def predict(
    score_network: ScoreNetwork,
    hypernetwork: HyperNetwork,
    topology_extractor: TopologyExtractor,
    data: np.ndarray,
    masks: np.ndarray,
    config: dict,
    device: torch.device,
    enable_tta: bool = True,
    train_data: np.ndarray = None,
    verbose: bool = True,
) -> dict:
    """Run score prediction pipeline.

    Args:
        score_network: Score network model.
        hypernetwork: Trained hypernetwork.
        topology_extractor: Topology extractor.
        data: Observed data (n, d).
        masks: Binary masks (n, d).
        config: Configuration dictionary.
        device: Torch device.
        enable_tta: Enable test-time adaptation.
        train_data: Training data for drift comparison (required for TTA).
        verbose: Print progress.

    Returns:
        Dictionary with prediction results.
    """
    # Initialize TTA adapter
    adapter = TestTimeAdapter(
        score_network=score_network,
        hypernetwork=hypernetwork,
        topology_extractor=topology_extractor,
        config=config,
        device=device,
    )

    # Set training topology if available
    if train_data is not None:
        adapter.set_training_topology(train_data)

    if enable_tta and train_data is not None:
        # Full TTA pipeline
        result = adapter.predict_with_tta(
            test_data_np=data * masks,
            test_masks_np=masks,
            verbose=verbose,
        )
    else:
        # No TTA: just hypernetwork initialization + forward pass
        observed = data * masks
        topo_result = topology_extractor.extract(observed)
        topo_features = torch.tensor(
            topo_result["features"], dtype=torch.float32, device=device
        )

        hypernetwork.eval()
        with torch.no_grad():
            generated_params = hypernetwork(topo_features)
        hypernetwork.apply_params_to_network(score_network, generated_params)

        score_network.eval()
        data_t = torch.tensor(observed, dtype=torch.float32, device=device)
        masks_t = torch.tensor(masks, dtype=torch.float32, device=device)

        with torch.no_grad():
            scores = score_network(data_t, masks_t)

        result = {
            "scores": scores.cpu().numpy(),
            "drift_info": None,
            "adapted": False,
        }

    return result


def main():
    """CLI entry point for prediction."""
    parser = argparse.ArgumentParser(
        description="Meta-TTA-TSM: Score Prediction"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Config file path."
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Checkpoint path."
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Input data path (.npy)."
    )
    parser.add_argument(
        "--masks", type=str, default=None, help="Masks file (.npy)."
    )
    parser.add_argument(
        "--train-data",
        type=str,
        default=None,
        help="Training data for TTA drift comparison (.npy).",
    )
    parser.add_argument(
        "--enable-tta", action="store_true", help="Enable test-time adaptation."
    )
    parser.add_argument(
        "--drift-threshold",
        type=float,
        default=None,
        help="Override drift threshold.",
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output scores path (.npy)."
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Merge training config if exists
    training_config_path = os.path.join(
        os.path.dirname(args.config), "training.yaml"
    )
    if os.path.exists(training_config_path):
        with open(training_config_path, "r") as f:
            config.update(yaml.safe_load(f))

    if args.drift_threshold is not None:
        config.setdefault("test_time_adaptation", {})[
            "drift_threshold"
        ] = args.drift_threshold

    # Setup
    set_seed(args.seed)
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )

    print(f"Device: {device}")

    # Load models
    score_network, hypernetwork, topology_extractor = load_checkpoint(
        args.checkpoint, config, device
    )

    # Load data
    data = np.load(args.input)
    if args.masks:
        masks = np.load(args.masks)
    else:
        masks = np.ones_like(data)

    train_data = np.load(args.train_data) if args.train_data else None

    print(f"Input shape: {data.shape}")
    print(f"TTA enabled: {args.enable_tta}")

    # Run prediction
    result = predict(
        score_network=score_network,
        hypernetwork=hypernetwork,
        topology_extractor=topology_extractor,
        data=data,
        masks=masks,
        config=config,
        device=device,
        enable_tta=args.enable_tta,
        train_data=train_data,
    )

    # Save results
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.save(args.output, result["scores"])
    print(f"Scores saved to: {args.output}")
    print(f"Score shape: {result['scores'].shape}")

    if result["drift_info"]:
        print(f"Topological drift: {result['drift_info']['drift']:.4f}")
        print(f"Adapted: {result['adapted']}")


if __name__ == "__main__":
    main()
