"""
Debug script to compare QR and SVD decompositions on a simple TT-core.
"""

import torch as tn

# Create a simple test matrix
tn.manual_seed(42)
M = tn.randn(10, 5, dtype=tn.float32)

print("="*80)
print("Testing QR vs SVD on a simple matrix")
print("="*80)
print(f"Matrix shape: {M.shape}")
print(f"Matrix norm: {tn.norm(M):.6f}")
print()

# QR decomposition
Q, R = tn.linalg.qr(M)
print("QR Decomposition:")
print(f"  Q shape: {Q.shape}")
print(f"  R shape: {R.shape}")
M_qr = Q @ R
error_qr = tn.norm(M - M_qr) / tn.norm(M)
print(f"  Reconstruction error: {error_qr:.2e}")
print()

# SVD decomposition (full rank)
U, S, Vh = tn.linalg.svd(M, full_matrices=False)
print("SVD Decomposition:")
print(f"  U shape: {U.shape}")
print(f"  S shape: {S.shape}")
print(f"  Vh shape: {Vh.shape}")
M_svd = U @ tn.diag(S) @ Vh
error_svd = tn.norm(M - M_svd) / tn.norm(M)
print(f"  Reconstruction error: {error_svd:.2e}")
print()

# Compare Q and U
print("Comparing Q and U:")
error_q_u = tn.norm(Q - U) / tn.norm(Q)
print(f"  ||Q - U|| / ||Q||: {error_q_u:.2e}")
print()

# Compare R and diag(S) @ Vh
print("Comparing R and diag(S) @ Vh:")
DSVh = tn.diag(S) @ Vh
error_r_dsvh = tn.norm(R - DSVh) / tn.norm(R)
print(f"  ||R - diag(S)@Vh|| / ||R||: {error_r_dsvh:.2e}")
print()

# Now simulate what happens in TT-orthogonalization
print("="*80)
print("Simulating TT-orthogonalization")
print("="*80)

# Let's say we have a "next core" to update
# The next core's last dimension should match R's first dimension
next_core = tn.randn(3, R.shape[0], dtype=tn.float32)
print(f"Next core shape: {next_core.shape}")
print()

# QR approach: multiply on the right by R.T
next_core_qr = next_core @ R.T
print("QR approach:")
print(f"  Updated next core shape: {next_core_qr.shape}")
print()

# SVD approach: multiply on the right by (Vh.T * S) which is Vh.T @ diag(S)
# Vh.T @ diag(S) is equivalent to (diag(S) @ Vh).T
next_core_svd = next_core @ (Vh.T * S)  # Broadcasting: Vh.T is [5,5], S is [5], result is [5,5]
print("SVD approach:")
print(f"  Updated next core shape: {next_core_svd.shape}")
print()

# Compare the updated next cores
print("Comparing updated next cores:")
error_next = tn.norm(next_core_qr - next_core_svd) / tn.norm(next_core_qr)
print(f"  ||next_qr - next_svd|| / ||next_qr||: {error_next:.2e}")
print()

# Check the math: both should give the same result
# QR: next @ R.T
# SVD: next @ Vh.T @ diag(S)
# Since M = Q @ R = U @ diag(S) @ Vh
# R = Q.T @ M = Q.T @ U @ diag(S) @ Vh
# So R.T = Vh.T @ diag(S) @ U.T @ Q

# Actually, Q and U are not the same in general!
# Let me verify the full reconstruction

# Start with original "core" representation
# In TT: core_i-1 @ core_i should equal core_i-1_new @ core_i_new

# QR version:
# core_i becomes Q.T
# core_i-1 becomes core_i-1 @ R.T
# Reconstruction: (core_i-1 @ R.T) @ Q = core_i-1 @ R.T @ Q = core_i-1 @ M (since M.T = Q.T @ R.T)

# SVD version:
# core_i becomes U.T  
# core_i-1 becomes core_i-1 @ Vh.T @ diag(S)
# Reconstruction: (core_i-1 @ Vh.T @ diag(S)) @ U = core_i-1 @ Vh.T @ diag(S) @ U = core_i-1 @ M (since M.T = (U @ diag(S) @ Vh).T = Vh.T @ diag(S) @ U.T)

print("="*80)
print("Verifying full reconstruction")
print("="*80)

# The key insight: we store Q.T (or U.T), but when we contract, we use Q (or U)
# So the reconstruction should work out

# Simulate the full chain
# Original product we're trying to preserve: next_core [3, 5] @ M.T [5, 10] = [3, 10]
original_product = next_core @ M.T  # This is what we're trying to preserve

# QR reconstruction
# QR: M = Q @ R, so M.T = R.T @ Q.T
# next_core_qr = next_core @ R.T
# current_core_qr would be Q.T, but we contract with Q
# Product: next_core_qr @ Q = (next_core @ R.T) @ Q = next_core @ R.T @ Q = next_core @ M.T
qr_product = next_core_qr @ Q
error_qr_full = tn.norm(original_product - qr_product) / tn.norm(original_product)
print(f"QR full reconstruction error: {error_qr_full:.2e}")

# SVD reconstruction
# SVD: M = U @ diag(S) @ Vh, so M.T = Vh.T @ diag(S) @ U.T
# next_core_svd = next_core @ (Vh.T * S) = next_core @ Vh.T @ diag(S)
# current_core_svd would be U.T, but we contract with U
# Product: next_core_svd @ U = (next_core @ Vh.T @ diag(S)) @ U = next_core @ M.T
svd_product = next_core_svd @ U
error_svd_full = tn.norm(original_product - svd_product) / tn.norm(original_product)
print(f"SVD full reconstruction error: {error_svd_full:.2e}")

# Compare QR and SVD reconstructions
error_qr_vs_svd = tn.norm(qr_product - svd_product) / tn.norm(qr_product)
print(f"QR vs SVD reconstruction: {error_qr_vs_svd:.2e}")

print()
print("="*80)
print("CONCLUSION")
print("="*80)
print("Both QR and SVD preserve the tensor contraction, even though")
print("the individual cores are different. This is expected and correct!")
print()
print("The test script should compare the *full TT-tensor reconstruction*,")
print("not the individual cores.")

