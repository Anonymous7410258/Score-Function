# Methodology

> Detailed description of the Meta-TTA-TSM algorithm and its components.

## Problem Setting

Let X ∈ R^d be a random variable with unknown density p_data(x). We observe n i.i.d. samples with missing data described by binary masks M ∈ {0,1}^d, where M_j = 1 indicates the j-th coordinate is observed. The goal is to learn the **complete-data score** ∇_x log p_data(x) using only partial observations under potentially shifting missingness patterns.

## Key Insight

Different missingness patterns induce topologically distinct observed data manifolds. Rather than treating missingness as purely statistical (importance weighting, imputation), this framework reasons about the **structural** changes to the data manifold using persistent homology, and adapts the score function accordingly.

## Algorithm: Meta-TTA-TSM

### Phase 1: Topological Feature Extraction

For each task T with observed data, compute:

1. **Persistence Diagrams**: Apply Vietoris-Rips filtration to the observed point cloud, producing persistence diagrams P_T = {P_T^(k)} for k = 0, ..., K (connected components, loops, etc.).

2. **Persistence Images**: Convert diagrams to fixed-size vectors via Gaussian KDE on a birth-persistence grid:
   ```
   f_T = Φ(P_T) ∈ R^m
   ```

### Phase 2: Hypernetwork-Based Meta-Learning

**Hypernetwork**: H_ϕ : R^m → Θ maps topological features to score network parameters:
```
θ_T^(0) = H_ϕ(f_T)
```

**Inner Loop** (K steps per task):
```
θ_T^(k+1) = θ_T^(k) - α_inner ∇_θ L_ISM(θ_T^(k))
```

where the ISM loss is:
```
L_ISM(θ) = E_x [ (1/2) ||s_θ(x)||² + tr(∇_x s_θ(x)) ]
```

**Outer Loop** (meta-objective):
```
L_meta(ϕ) = E_T [ L_ISM(θ_T^(K)(ϕ)) ]
ϕ ← ϕ - α_meta ∇_ϕ L_meta(ϕ)
```

### Phase 3: Test-Time Adaptation

1. **Drift Detection**: Compute Wasserstein distance between training and test persistence diagrams:
   ```
   Δ_topo = d_W(P_train, P_test)
   ```

2. **Adaptation** (if Δ_topo > τ): Minimize combined objective:
   ```
   L_adapt(θ) = L_ISM(θ; D_test) + λ_topo · L_topo(θ; P_test)
   ```
   where L_topo measures topological consistency (Wasserstein distance between generated and target persistence diagrams).

## Theoretical Guarantees

1. **Meta-Learning Convergence** (Theorem 4.1): The meta-objective converges at rate O(1/√T).
2. **Inner-Loop Stability** (Lemma 4.2): Hypernetwork initialization controls adaptation error.
3. **TTA Stability** (Theorem 4.3): Adaptation improves proportionally to topological drift magnitude.
4. **Task Complexity** (Theorem 4.4): Topological diversity governs generalization, not just task count.

## Implementation Notes

- **Hutchinson Trace Estimator**: The Jacobian trace in the ISM loss is estimated stochastically using random Rademacher vectors for efficiency in high dimensions.
- **Chunked Parameter Generation**: The hypernetwork generates score network parameters layer-by-layer to manage parameter space size.
- **Spectral Normalization**: Applied to the hypernetwork to enforce the Lipschitz constraint required by theoretical guarantees.
- **Persistence Image Vectorization**: Provides a differentiable, stable approximation to persistence diagrams for gradient-based optimization.
