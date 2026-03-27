# Experiments

> Experimental setup, hyperparameters, and expected results.

## Datasets

| Dataset | Type | Dimensions | Samples | Description |
|---|---|---|---|---|
| Truncated Gaussian | Synthetic | 20–100 | 5,000 | Multivariate Gaussian truncated to positive orthant |
| Non-truncated Gaussian | Synthetic | 20–100 | 5,000 | Standard multivariate Gaussian |
| ICA Non-Gaussian | Synthetic | 20–50 | 5,000 | Independent Component Analysis with Laplacian/uniform sources |
| Gaussian Graphical Model | Structured | 20–100 | 5,000 | Sparse precision matrix model |
| Financial Time-Series | Real-world | 50–200 | 5,000 | Correlated factor model with heavy tails |
| Biological Expression | Real-world | 100–1000+ | 2,000 | Low-rank sparse gene expression proxy |

## Missingness Regimes

- **MCAR rates**: 20%, 30%, 40%, 50%, 60%, 70%, 80%, 95%
- **Shift scenarios**: Training at rate r, testing at rate r' where r ≠ r'
- **Structural shift**: Block-wise vs. element-wise missingness

## Evaluation Metrics

| Metric | Description | Used For |
|---|---|---|
| Fisher Divergence | Primary: ||s_θ(x) - ∇log p(x)||² | All experiments |
| MMD | Distribution-level comparison | Quality assessment |
| NLL | Negative log-likelihood | Gaussian experiments |
| AUC-ROC | Area under ROC curve | GGM structure recovery |
| SHD | Structural Hamming Distance | GGM structure recovery |

## Hyperparameters

### Default Configuration

| Parameter | Value | Notes |
|---|---|---|
| Meta-LR (α_meta) | 1e-3 | Adam optimizer with cosine schedule |
| Inner-LR (α_inner) | 1e-2 | SGD for inner loop |
| Inner steps (K) | 5 | MAML-style adaptation |
| TTA-LR (α_tta) | 5e-3 | SGD for test-time adaptation |
| TTA steps (K_adapt) | 10 | Adaptation gradient steps |
| λ_topo | 0.1 | Topological consistency weight |
| Drift threshold (τ) | 0.5 | Wasserstein distance threshold |
| Tasks per batch | 8 | Meta-batch size |
| Meta-epochs | 200 | Total training epochs |
| Score network | [256, 256, 128] | Hidden layer sizes |
| Hypernetwork | [512, 512] | Hidden layer sizes |
| Persistence image | 10 × 10 | Resolution per dimension |
| Max homology dim | 1 | H0 and H1 features |

### Results averaged over 5 random seeds with standard deviations.

## Main Results

### Table 3: Fisher Divergence on Gaussian Models

| Method | 20% Missing | 40% Missing | 60% Missing | 80% Missing |
|---|---|---|---|---|
| Full-data (oracle) | 0.023 ± 0.001 | 0.023 ± 0.001 | 0.023 ± 0.001 | 0.023 ± 0.001 |
| Zeroed | 0.145 ± 0.008 | 0.312 ± 0.015 | 0.678 ± 0.032 | 1.456 ± 0.089 |
| EM | 0.089 ± 0.005 | 0.167 ± 0.009 | 0.445 ± 0.024 | 1.123 ± 0.067 |
| VAE-impute | 0.078 ± 0.004 | 0.156 ± 0.008 | 0.423 ± 0.023 | 1.089 ± 0.065 |
| Marg-IW | 0.067 ± 0.004 | 0.134 ± 0.007 | 0.356 ± 0.019 | 0.891 ± 0.052 |
| Marg-Var | 0.071 ± 0.004 | 0.142 ± 0.008 | 0.348 ± 0.018 | 0.878 ± 0.051 |
| **Meta-TTA-TSM** | **0.045 ± 0.002** | **0.078 ± 0.004** | **0.167 ± 0.009** | **0.334 ± 0.018** |
| **Improvement** | **33.0%** | **41.8%** | **52.0%** | **62.0%** |

## Reproducing Results

```bash
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Train on Gaussian data
python -m training.train \
    --config configs/config.yaml \
    --dataset gaussian \
    --missing-rate 0.4 \
    --seed 42

# Step 3: Evaluate across missingness rates
python -m evaluation.evaluate \
    --config configs/config.yaml \
    --checkpoint checkpoints/best_model.pt \
    --dataset gaussian \
    --missing-rates 0.2 0.4 0.6 0.8 \
    --n-seeds 5

# Step 4: View TensorBoard logs
tensorboard --logdir runs/
```

## Ablation Studies

The paper evaluates the following ablations:

1. **No topology** (–Topo): Replace topological features with random vectors → validates topology contribution
2. **No meta-learning** (–Meta): Train a single score network → validates meta-learning framework
3. **No TTA** (–TTA): Disable test-time adaptation → validates adaptation contribution
4. **Full model**: All components enabled → best performance
