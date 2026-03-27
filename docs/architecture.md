# System Architecture

> This document describes the architecture of the Meta-Topological Test-Time Adaptive Score Matching (Meta-TTA-TSM) framework.

## Overview

Meta-TTA-TSM addresses the problem of score function estimation under missing data with shifting missingness patterns. The architecture integrates three key components:

1. **Topological Feature Extraction** — Persistent homology captures the structural changes in data manifolds caused by different missingness patterns.
2. **Hypernetwork-Based Meta-Learning** — A hypernetwork maps topological features to score network parameters, learning transferable priors across diverse topologies.
3. **Test-Time Adaptation** — Wasserstein-based drift detection triggers principled adaptation at deployment.

## Component Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                        INPUT LAYER                               │
│  ┌────────────┐    ┌───────────────┐    ┌──────────────────┐    │
│  │ Raw Data   │───▶│ Missingness   │───▶│ Observed Data    │    │
│  │ x ∈ R^d    │    │ Mask M        │    │ x_Λ = x * M     │    │
│  └────────────┘    └───────────────┘    └──────┬───────────┘    │
└────────────────────────────────────────────────┬─────────────────┘
                                                 │
┌────────────────────────────────────────────────▼─────────────────┐
│                  TOPOLOGY EXTRACTION                             │
│  ┌────────────┐    ┌───────────────┐    ┌──────────────────┐    │
│  │ Point      │───▶│ Vietoris-Rips │───▶│ Persistence      │    │
│  │ Cloud      │    │ Filtration    │    │ Diagrams P_T     │    │
│  └────────────┘    └───────────────┘    └──────┬───────────┘    │
│                                                │                 │
│                                       ┌────────▼──────────┐     │
│                                       │ Persistence Images│     │
│                                       │ f_T = Φ(P_T)      │     │
│                                       └────────┬──────────┘     │
└────────────────────────────────────────────────┬─────────────────┘
                                                 │
┌────────────────────────────────────────────────▼─────────────────┐
│                    MODEL LAYER                                   │
│  ┌────────────────┐              ┌──────────────────────┐       │
│  │ Hypernetwork   │─────────────▶│ Score Network        │       │
│  │ H_ϕ: R^m → Θ  │   θ = H(f)  │ s_θ: R^d → R^d      │       │
│  └────────────────┘              └──────────────────────┘       │
└──────────────────────────────────────────────────────────────────┘
                                                 │
┌────────────────────────────────────────────────▼─────────────────┐
│                 OPTIMIZATION LAYER                               │
│                                                                  │
│  Meta-Training:                                                  │
│    Outer: ϕ ← ϕ - α_meta ∇_ϕ L_meta(ϕ)                        │
│    Inner: θ ← θ - α_inner ∇_θ L_ISM(θ)  × K steps             │
│                                                                  │
│  Test-Time Adaptation:                                           │
│    θ ← θ - α_tta ∇_θ [L_ISM + λ L_topo]  × K_adapt steps      │
└──────────────────────────────────────────────────────────────────┘
```

## Module Dependencies

```
models/
├── score_network.py    ← Core neural network
├── hypernetwork.py     ← Depends on score_network
├── topology.py         ← Independent (uses numpy/scipy)
└── losses.py           ← Depends on score_network

data/
├── datasets.py         ← Independent
└── missingness.py      ← Independent

training/
├── meta_trainer.py     ← Depends on models/*, data/*
└── train.py            ← Depends on meta_trainer, data

inference/
├── tta.py              ← Depends on models/*
└── predict.py          ← Depends on tta, models

evaluation/
├── metrics.py          ← Independent (numpy/sklearn)
└── evaluate.py         ← Depends on inference, data, metrics
```

## Data Flow

1. **Data Generation**: `data/datasets.py` generates samples from specified distributions.
2. **Mask Application**: `data/missingness.py` produces MCAR masks and creates tasks.
3. **Topology Extraction**: `models/topology.py` computes persistence diagrams and images.
4. **Parameter Generation**: `models/hypernetwork.py` maps topology features → score params.
5. **Score Estimation**: `models/score_network.py` estimates ∇ log p(x).
6. **Training**: `training/meta_trainer.py` runs outer/inner loop optimization.
7. **Adaptation**: `inference/tta.py` detects drift and adapts at test time.
8. **Evaluation**: `evaluation/metrics.py` computes Fisher divergence, MMD, etc.
