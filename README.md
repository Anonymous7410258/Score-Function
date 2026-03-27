# Meta-Topological Test-Time Adaptive Score Matching (Meta-TTA-TSM)

> This repository implements the method proposed in the accompanying research paper: *"Meta-Topological Test-Time Adaptive Score Matching"*.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Abstract

Standard score matching methods assume the data manifold is fully observed and structurally stable — assumptions that fail under missing data. This work introduces **Meta-Topological Test-Time Adaptive Score Matching (Meta-TTA-TSM)**, a framework that integrates persistent homology with meta-learning to produce score functions that are dynamically conditioned on the topology of partially observed data. A hypernetwork maps topological features — extracted via persistence images — to score network parameters, enabling rapid adaptation to novel missingness patterns. At test time, Wasserstein distance on persistence diagrams detects topological drift, triggering principled adaptation that preserves both score matching fidelity and topological consistency.

## Key Contributions

1. **Topology-Conditioned Score Functions**: First framework to use persistent homology features to parameterize score networks via a hypernetwork, enabling topology-aware estimation under missing data.
2. **Meta-Learning Over Topological Tasks**: Formulates diverse missingness patterns as a distribution over topological tasks, allowing generalization across structural distribution shifts.
3. **Test-Time Adaptation Under Topological Drift**: Detects and adapts to novel topological structures at deployment via Wasserstein-based drift detection and principled topological consistency regularization.
4. **Theoretical Guarantees**: Convergence guarantees for meta-learning and stability bounds for test-time adaptation under bounded topological shift.
5. **State-of-the-Art Results**: 33–62% improvement in Fisher divergence over best baselines across synthetic, non-Gaussian, graphical, and real-world datasets.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    META-TRAINING PHASE                              │
│                                                                     │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────┐               │
│  │ Observed  │───▶│  Persistent  │───▶│  Persistence │               │
│  │ Data x_Λ  │    │  Homology    │    │  Images f_T  │               │
│  └──────────┘    └──────────────┘    └──────┬───────┘               │
│                                             │                       │
│                                    ┌────────▼────────┐              │
│                                    │  Hypernetwork    │              │
│                                    │  H_ϕ: R^m → Θ   │              │
│                                    └────────┬────────┘              │
│                                             │  θ_T^(0)              │
│                                    ┌────────▼────────┐              │
│                                    │  Score Network   │              │
│                                    │  s_θ: R^d → R^d  │              │
│                                    └────────┬────────┘              │
│                                             │                       │
│  ┌──────────────────────────────────────────▼──────────────────┐    │
│  │  Inner-Loop: K gradient steps on ISM loss (per task)        │    │
│  │  Outer-Loop: Meta-gradient on hypernetwork ϕ                │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    TEST-TIME ADAPTATION                              │
│                                                                     │
│  1. Compute persistence diagram P_test from test data               │
│  2. Measure drift: Δ_topo = W_p(P_train, P_test)                   │
│  3. If drift > τ: adapt θ via L_ISM + λ_topo · L_topo              │
└─────────────────────────────────────────────────────────────────────┘
```

## Repository Structure

```
meta-tta-tsm/
├── README.md                   # This file
├── LICENSE                     # MIT License
├── requirements.txt            # Python dependencies
├── environment.yml             # Conda environment
├── setup.py                    # Package installation
├── CITATION.cff                # Citation metadata
├── .gitignore                  # Git ignore rules
│
├── configs/
│   ├── config.yaml             # Master configuration
│   └── training.yaml           # Training hyperparameters
│
├── models/
│   ├── __init__.py
│   ├── score_network.py        # Score network s_θ
│   ├── hypernetwork.py         # Hypernetwork H_ϕ
│   ├── topology.py             # Persistent homology + features
│   └── losses.py               # ISM + topological losses
│
├── data/
│   ├── __init__.py
│   ├── datasets.py             # Dataset generators & loaders
│   └── missingness.py          # MCAR mask generation
│
├── training/
│   ├── __init__.py
│   ├── train.py                # Training entry point
│   └── meta_trainer.py         # Meta-learning trainer
│
├── inference/
│   ├── __init__.py
│   ├── predict.py              # Score prediction
│   └── tta.py                  # Test-time adaptation
│
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py              # Fisher div, MMD, NLL, AUC, SHD
│   └── evaluate.py             # Evaluation pipeline
│
├── utils/
│   ├── __init__.py
│   ├── logger.py               # TensorBoard + console logging
│   ├── preprocessing.py        # Data normalization
│   └── reproducibility.py      # Seed & determinism
│
├── scripts/
│   ├── run_training.sh         # Training script
│   ├── run_evaluation.sh       # Evaluation script
│   └── download_dataset.sh     # Dataset download
│
├── docs/
│   ├── architecture.md         # System architecture
│   ├── methodology.md          # Algorithm details
│   └── experiments.md          # Experimental setup & results
│
└── tests/
    └── test_model.py           # Unit tests
```

## Installation

### Option 1: pip

```bash
git clone https://github.com/anonymous/meta-tta-tsm.git
cd meta-tta-tsm
pip install -r requirements.txt
pip install -e .
```

### Option 2: Conda

```bash
git clone https://github.com/anonymous/meta-tta-tsm.git
cd meta-tta-tsm
conda env create -f environment.yml
conda activate meta-tta-tsm
pip install -e .
```

## Dataset Preparation

This repository generates synthetic datasets programmatically. Supported datasets:

| Dataset | Type | Dimensions |
|---|---|---|
| Truncated Gaussian | Synthetic | 20–100 |
| Non-truncated Gaussian | Synthetic | 20–100 |
| ICA Non-Gaussian | Synthetic | 20–50 |
| Gaussian Graphical Model | Structured | 20–100 |
| Financial Time-Series | Real-world | 50–200 |
| Biological Expression | Real-world | 100–1000+ |

For real-world datasets, place them under `data/raw/` and use the preprocessing pipeline:

```bash
python -m utils.preprocessing --input data/raw/ --output data/processed/
```

## Training

### Meta-Training

```bash
python -m training.train \
    --config configs/config.yaml \
    --dataset gaussian \
    --missing-rate 0.4 \
    --device cuda \
    --seed 42
```

### Using Shell Script

```bash
bash scripts/run_training.sh
```

### Key Hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `meta_lr` | 1e-3 | Meta-learning rate (outer loop) |
| `inner_lr` | 1e-2 | Inner-loop learning rate |
| `inner_steps` | 5 | Inner-loop gradient steps (K) |
| `lambda_topo` | 0.1 | Topological consistency weight |
| `num_meta_epochs` | 200 | Number of meta-training epochs |
| `tasks_per_batch` | 8 | Tasks per meta-batch |

## Evaluation

```bash
python -m evaluation.evaluate \
    --config configs/config.yaml \
    --checkpoint checkpoints/best_model.pt \
    --dataset gaussian \
    --missing-rates 0.2 0.4 0.6 0.8 \
    --device cuda
```

### Using Shell Script

```bash
bash scripts/run_evaluation.sh
```

## Inference

Run score prediction on new data with optional test-time adaptation:

```bash
python -m inference.predict \
    --config configs/config.yaml \
    --checkpoint checkpoints/best_model.pt \
    --input data/processed/test_data.npy \
    --enable-tta \
    --drift-threshold 0.5 \
    --output results/scores.npy
```

## Expected Results

### Fisher Divergence on Gaussian Models (Lower is Better)

| Method | 20% Missing | 40% Missing | 60% Missing | 80% Missing |
|---|---|---|---|---|
| Marg-IW (best baseline) | 0.067 | 0.134 | 0.356 | 0.891 |
| **Meta-TTA-TSM (ours)** | **0.045** | **0.078** | **0.167** | **0.334** |
| **Improvement** | **33.0%** | **41.8%** | **52.0%** | **62.0%** |

Results are averaged over 5 random seeds. See the paper for complete results across all datasets.

## Monitoring

Training logs are written to TensorBoard:

```bash
tensorboard --logdir runs/
```

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{anonymous2025metatta,
  title={Meta-Topological Test-Time Adaptive Score Matching},
  author={Anonymous},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2025}
}
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
