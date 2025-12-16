"""L=2 Tensor Basis Matrices for Spin Hamiltonian Terms.

This module defines the basis matrices used to construct the spin-spin interaction
tensors in SpinGNN++. The general spin Hamiltonian can be written as:

    H = Σ_{ij} S_i^T J_ij S_j + Σ_i S_i^T A_i S_i

where J_ij is the 3x3 exchange tensor and A_i is the 3x3 on-site anisotropy tensor.

=== EXCHANGE TENSOR DECOMPOSITION (J_ij) ===

The exchange tensor J_ij can be decomposed into irreducible representations:

1. ISOTROPIC (L=0, scalar, 1 DOF):
   - term_H: Heisenberg exchange J * I_3x3
   - Contributes: J * (S_i · S_j)

2. ANTISYMMETRIC (L=1, pseudovector, 3 DOF):
   - term_DM1, term_DM2, term_DM3: Dzyaloshinskii-Moriya (DM) interaction
   - Generates cross-product: D · (S_i × S_j)
   - D_x -> term_DM1, D_y -> term_DM2, D_z -> term_DM3

3. SYMMETRIC TRACELESS (L=2, quadrupole, 5 DOF):
   - term_ASEI1-5: Anisotropic Symmetric Exchange Interaction
   - These form a basis for symmetric traceless 3x3 matrices
   - Total: 5 independent components

Total for J_ij: 1 + 3 + 5 = 9 components (full 3x3 tensor)

=== ANISOTROPY TENSOR DECOMPOSITION (A_i) ===

The on-site anisotropy tensor A_i must be symmetric (from energy minimization):

1. ISOTROPIC (L=0): term_H (but often ignored as constant shift)
2. SYMMETRIC TRACELESS (L=2): term_ASEI1-5

Total for A_i: 1 + 5 = 6 components (symmetric 3x3 tensor)

=== MAPPING FROM IRREPS ===

The MSENN network outputs equivariant features with irreps 0e + 1e + 2e:
- 0e (1 component)  -> term_H (Heisenberg)
- 1e (3 components) -> term_DM1-3 (DM vector)  [only for J, not A]
- 2e (5 components) -> term_ASEI1-5 (symmetric traceless)

These features are contracted with the basis matrices to form the full tensors.
"""

import torch


# =============================================================================
# HEISENBERG EXCHANGE (L=0, isotropic)
# =============================================================================
# Identity matrix: contributes J * (S_i · S_j)
term_H = torch.eye(3)

# =============================================================================
# DZYALOSHINSKII-MORIYA INTERACTION (L=1, antisymmetric)
# =============================================================================
# These antisymmetric matrices encode the DM vector D = (D_x, D_y, D_z)
# The DM interaction is: D · (S_i × S_j) = S_i^T (D × I) S_j
# where (D × I) is the antisymmetric matrix formed from D

# DM component along x-axis: generates (S_i × S_j)_x contribution
term_DM1 = torch.tensor([[0., 0.,  0.], 
                         [0., 0.,  1.], 
                         [0., -1., 0.]])

# DM component along y-axis: generates (S_i × S_j)_y contribution
term_DM2 = torch.tensor([[0., 0.,  -1.], 
                         [0., 0.,  0.], 
                         [1., 0., 0.]])

# DM component along z-axis: generates (S_i × S_j)_z contribution
term_DM3 = torch.tensor([[0., 1.,  0.], 
                         [-1., 0.,  0.], 
                         [0., 0., 0.]])

# =============================================================================
# ANISOTROPIC SYMMETRIC EXCHANGE INTERACTION (L=2, symmetric traceless)
# =============================================================================
# These 5 symmetric traceless matrices form a complete basis for L=2 tensors
# They correspond to the 5 spherical harmonics Y_2^m

# ASEI1: Corresponds to (xx - yy) component, like Y_2^2
term_ASEI1 = torch.tensor([[1., 0.,  0.], 
                           [0., -1.,  0.], 
                           [0., 0., 0.]])

# ASEI2: Corresponds to xy component, like Y_2^{-2}
term_ASEI2 = torch.tensor([[0., 1.,  0.], 
                           [1., 0.,  0.], 
                           [0., 0., 0.]])

# ASEI3: Corresponds to xz component, like Y_2^1
term_ASEI3 = torch.tensor([[0., 0.,  1.], 
                           [0., 0.,  0.], 
                           [1., 0., 0.]])

# ASEI4: Corresponds to yz component, like Y_2^{-1}
term_ASEI4 = torch.tensor([[0., 0.,  0.], 
                           [0., 0.,  1.], 
                           [0., 1., 0.]])

# ASEI5: Corresponds to (2zz - xx - yy) component, like Y_2^0
# Note: traceless since -1 + (-1) + 1 + 1 = 0 when properly normalized
term_ASEI5 = torch.tensor([[0., 0.,  0.], 
                           [0., -1.,  0.], 
                           [0., 0., 1.]])


# =============================================================================
# ASSEMBLED TENSOR BASES
# =============================================================================

# Full J_ij tensor basis: 9 matrices (0e + 1e + 2e = 1 + 3 + 5)
# Order: [Heisenberg, DM_x, DM_y, DM_z, ASEI_1, ASEI_2, ASEI_3, ASEI_4, ASEI_5]
matrix_terms_list_J = [term_H, term_DM1, term_DM2, term_DM3,
                       term_ASEI1, term_ASEI2, term_ASEI3, term_ASEI4, term_ASEI5]

# Symmetric A_i tensor basis: 6 matrices (0e + 2e = 1 + 5)
# Order: [Heisenberg, ASEI_1, ASEI_2, ASEI_3, ASEI_4, ASEI_5]
# Note: No DM terms since A_i must be symmetric
matrix_terms_list_A = [term_H, term_ASEI1, term_ASEI2, term_ASEI3, term_ASEI4, term_ASEI5]

# Stack into tensors for efficient batched contraction
# Shape: [num_terms, 3, 3]
matrix_terms_J = torch.concat([el.unsqueeze(0) for el in matrix_terms_list_J])
matrix_terms_A = torch.concat([el.unsqueeze(0) for el in matrix_terms_list_A])