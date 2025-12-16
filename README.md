# SpinGNN / SpinGNN++

Machine learning interatomic potentials for magnetic materials, extending the [Allegro](https://github.com/mir-group/allegro) architecture to handle spin degrees of freedom.

## Overview

SpinGNN++ is an equivariant graph neural network that predicts the total energy of magnetic systems by decomposing it into physically motivated terms:

```
E = E_pair + E_BQ + E_J + E_A + E_TENN
```

| Term | Description | Physics |
|------|-------------|---------|
| E_pair | Standard pair energy | Non-magnetic interactions |
| E_BQ | Biquadratic exchange | K(S_i · S_j)² |
| E_J | Exchange tensor | S_i^T J_ij S_j (Heisenberg + DM + anisotropic) |
| E_A | On-site anisotropy | S_i^T A_i S_i |
| E_TENN | TENN contribution | Higher-order spin-lattice coupling |

## Changes from Original Allegro

This codebase extends the original Allegro architecture with the following additions:

### New Modules - SpinGNN++

| File | Description |
|------|-------------|
| `_spin_embedding.py` | Spin distance encoding and TENN spherical harmonics |
| `_SpinGNNPlus.py` (nn) | MSENN and TENN Allegro variants |
| `_edgewise_MSENN.py` | Energy sum modules for J, A, K, TENN terms |
| `l2_matrix.py` | L=2 tensor basis for spin Hamiltonian |
| `_SpinGNNPlus.py` (model) | Model builder for SpinGNN++ |

### New Modules - ETN (Equivariant Tensor Network)

| File | Description |
|------|-------------|
| `_etn.py` | Core ETN module with tensor train decomposition |
| `_etn_opt.py` | Optimized ETN with FX graph compilation |
| `_etn_als_opt.py` | ETN with ALS (Alternating Least Squares) optimization |
| `_edge_features_F.py` | Edge feature construction for ETN |
| `_strided/_contract_ETN.py` | Optimized tensor contraction for ETN |
| `model/_ETN.py` | ETN model builder |

### Extended Modules

| File | Changes |
|------|---------|
| `_keys.py` | Added spin-related and ETN field keys (40+ new keys) |
| `_allegro.py` | Added `Allegro_Module_SEGNN` with spin inputs |
| `_strided/_channels.py` | Added `MakeWeightedChannelsTENN` for 3-channel input |
| `nn/__init__.py` | Exports for new modules |

## Architectures

This codebase provides two distinct equivariant architectures:

1. **SpinGNN++**: Allegro-based architecture extended for magnetic materials
2. **ETN**: Tensor train-based architecture with Wigner-3j coupling

---

## ETN (Equivariant Tensor Network)

ETN is a fundamentally different architecture from Allegro that uses tensor train decomposition with explicit Wigner-3j coupling.

### ETN vs Allegro

| Aspect | Allegro/SpinGNN++ | ETN |
|--------|-------------------|-----|
| Representation | Edge-based | Node-based (aggregated from edges) |
| Core operation | Message passing + tensor products | Tensor train contraction |
| Coupling | Learned weights | Explicit Wigner-3j symbols |
| Body order | Implicit via layers | Explicit via depth parameter `d` |

### ETN Architecture

```
Input: positions, atom_types
            |
            v
    +-------------------+
    | Pair Type Embed   |  <-- Encode (type_i, type_j) pairs
    +-------------------+
            |
            v
    +-------------------+
    | Radial Basis Q    |  <-- Distance encoding
    +-------------------+
            |
            v
    +-------------------+
    | Spherical Harm. Y |  <-- Angular encoding
    +-------------------+
            |
            v
    +-------------------+
    | EdgeFeatures_F    |  <-- F = Y * (B * A * Q)
    +-------------------+
            |
            v
    +-------------------+
    | EdgewiseFSum      |  <-- Aggregate edges → nodes
    +-------------------+
            |
            v
    +-------------------+
    | ETN_Module        |  <-- Tensor train contraction
    +-------------------+
            |
            v
       Per-atom Energy
```

### Tensor Train Structure

ETN parameterizes a high-order equivariant tensor using tensor train format:

```
T = core2_1 × core3[0] × core3[1] × ... × core3[d-3] × core2_d
```

where:
- `core2_1`, `core2_d`: Second-order boundary cores (matrices per l)
- `core3[i]`: Third-order cores with Wigner-3j coupling

The depth `d` controls body order:
- d=2: 2-body (pair interactions)
- d=3: 3-body
- d=4: 4-body

### ETN Configuration

```yaml
# ETN-specific parameters
Nc: 16                    # Number of feature channels
d: 4                      # Tensor train depth (body order)
N_rank_ett: [4, 8, 8, 4]  # TT ranks
N_rank_spec: 4            # Species embedding rank
l_max: 2                  # Maximum angular momentum
```

---

## SpinGNN++ Architecture

### MSENN Branch (Position-Only)

The Magnetic Steerable E(3) Neural Network processes only positional information but outputs equivariant features for spin Hamiltonian tensors:

```
Positions → Spherical Harmonics → Allegro_MSENN → [latent, J_features, A_features]
                                        ↓
                                   E_pair, E_BQ
```

**Key difference from standard Allegro:** Output irreps are `0e + 1e + 2e` (9 components) instead of just `0e` (scalars).

### TENN Branch (Position + Spin)

The Time-reversal Equivariant Tensor Network handles both position AND spin vectors with proper symmetry:

```
Positions + Spins → SphericalHarmonic_TENN → Allegro_TENN → E_TENN
                           ↓
              3 channels: [Y(r_ij), Y(S_i), Y(S_j)]
```

**Key innovation:** Uses extended irreps `(l, p, t)` where `t = ±1` indicates behavior under time reversal.

### Tensor Construction

Exchange and anisotropy tensors are built from learned features contracted with basis matrices:

```python
# Exchange tensor J_ij (9 components: Heisenberg + DM + anisotropic)
J_ij = Σ_k features_J[k] * matrix_terms_J[k]
E_J = S_i^T @ J_ij @ S_j

# Anisotropy tensor A_i (6 components: isotropic + anisotropic, no DM)
A_i = Σ_k features_A[k] * matrix_terms_A[k]  
E_A = S_i^T @ A_i @ S_i
```

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         INPUTS                                   │
│   Positions (r)    Atom Types (Z)    Spins (S)                  │
└────────┬─────────────────┬─────────────────┬────────────────────┘
         │                 │                 │
         ▼                 ▼                 │
┌─────────────────┐ ┌─────────────┐          │
│ Radial Basis    │ │ One-Hot     │          │
│ Encoding        │ │ Encoding    │          │
└────────┬────────┘ └──────┬──────┘          │
         │                 │                 │
         ▼                 │                 │
┌─────────────────┐        │                 │
│ Spherical       │        │                 │
│ Harmonics Y(r)  │        │                 │
└────────┬────────┘        │                 │
         │                 │                 │
         ▼                 ▼                 │
┌─────────────────────────────────┐          │
│        Allegro_MSENN            │          │
│   (Position-only processing)    │          │
└──┬──────────┬──────────┬────────┘          │
   │          │          │                   │
   ▼          ▼          ▼                   │
E_pair    J_features  A_features             │
   │          │          │                   │
   │          ▼          ▼                   ▼
   │    ┌─────────────────────────────────────────┐
   │    │         Spin Processing                  │
   │    │  • Spin distance: S_i · S_j             │
   │    │  • Radial basis encoding                │
   │    │  • Spherical harmonics of spins         │
   │    └──────────────┬──────────────────────────┘
   │                   │
   │          ┌────────┴────────┐
   │          ▼                 ▼
   │    ┌───────────┐    ┌─────────────────┐
   │    │ J, A, BQ  │    │  Allegro_TENN   │
   │    │ Energies  │    │  (Spin-aware)   │
   │    └─────┬─────┘    └───────┬─────────┘
   │          │                  │
   │          ▼                  ▼
   │       E_J, E_A           E_TENN
   │       E_BQ                  │
   │          │                  │
   └──────────┴────────┬─────────┘
                       ▼
              ┌────────────────┐
              │  Total Energy  │
              │  (weighted sum)│
              └────────────────┘
```

## Time-Reversal Symmetry

A key innovation in SpinGNN++ is proper handling of time-reversal symmetry. Under time reversal T:

| Quantity | Transformation |
|----------|---------------|
| Position r | r → r (invariant) |
| Momentum p | p → -p (flip) |
| Spin S | S → -S (flip) |

The TENN branch uses extended irreps `(l, p, t)` where:
- `l`: angular momentum
- `p`: parity under spatial inversion (±1)
- `t`: behavior under time reversal (±1)

This allows the network to learn magnetic interactions while respecting fundamental symmetries.

## Spin Hamiltonian Basis

The `l2_matrix.py` module defines basis matrices for spin interactions:

### Exchange Tensor J_ij (9 components)

| Component | Irrep | Physics | Matrix |
|-----------|-------|---------|--------|
| term_H | 0e | Heisenberg J(S·S) | I₃ |
| term_DM1-3 | 1e | DM interaction D·(S×S) | Antisymmetric |
| term_ASEI1-5 | 2e | Anisotropic exchange | Symmetric traceless |

### Anisotropy Tensor A_i (6 components)

| Component | Irrep | Physics |
|-----------|-------|---------|
| term_H | 0e | Isotropic |
| term_ASEI1-5 | 2e | Anisotropic (easy-axis, easy-plane, etc.) |

Note: No 1e (DM) terms in A_i because on-site anisotropy must be symmetric.

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/SpinGNN.git
cd SpinGNN

# Install nequip (local)
cd nequip
pip install -e .
cd ..

# Install allegro with SpinGNN extensions (local)
cd allegro
pip install -e .
cd ..
```

## Requirements

- Python >= 3.8
- PyTorch >= 1.11
- e3nn >= 0.5.0
- NequIP (included in repo)

## Usage

### Configuration

See `allegro/workdir/configs/example_SpinGNNPlus.yaml` for a complete example configuration.

Key parameters:
```yaml
# Model architecture
l_max: 2                      # Maximum angular momentum
parity: o3_full               # Full O(3) with both parities
num_layers: 2                 # Depth of Allegro networks
env_embed_multiplicity: 32    # Width of environment embeddings

# For TENN
irreps_edge_sh_TENN: "1x0e + 1x1e + 1x2e"  # With time-reversal
```

### Training

```bash
nequip-train configs/example_SpinGNNPlus.yaml
```

### Data Format

Input data should include:
- `pos`: Atomic positions [N, 3]
- `atom_types`: Atom type indices [N]
- `node_spin`: Spin vectors [N, 3]
- `energy`: Total energy (scalar)
- `forces`: Forces [N, 3] (optional)

## Citation

Also cite the original Allegro paper:
```bibtex
@article{musaelian2023learning,
  title={Learning local equivariant representations for scalable molecular dynamics},
  author={Musaelian, Albert and Batzner, Simon and Jober, Anders and others},
  journal={Nature Communications},
  year={2023}
}
```

## License

This project extends Allegro which is licensed under the MIT License.

## Acknowledgments

- Original Allegro implementation by the [MIR Group](https://github.com/mir-group/allegro)
- NequIP framework by the [MIR Group](https://github.com/mir-group/nequip)
- e3nn library for equivariant neural networks
