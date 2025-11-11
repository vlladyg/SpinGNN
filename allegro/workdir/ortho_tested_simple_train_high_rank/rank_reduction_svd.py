"""
SVD-based rank reduction for TT-cores with equivariant structure.

This module provides an alternative to QR-based orthogonalization that can
actually reduce the rank of TT-cores using SVD truncation.
"""

import torch as tn


def SVD(mat, rank=None, threshold=None):
    """
    Compute the SVD decomposition with optional rank truncation.
    
    Parameters
    ----------
    mat : torch.Tensor
        Matrix to decompose
    rank : int, optional
        Target rank for truncation. If None, no truncation.
    threshold : float, optional
        Threshold for singular value truncation. If None, no threshold.
        
    Returns
    -------
    U : torch.Tensor
        Left singular vectors
    S : torch.Tensor
        Singular values
    Vh : torch.Tensor
        Right singular vectors (already transposed/conjugated)
    """
    U, S, Vh = tn.linalg.svd(mat, full_matrices=False)
    
    # Determine actual rank to keep
    if rank is not None and threshold is not None:
        # Keep at most 'rank' singular values above threshold
        mask = S > threshold
        keep = min(rank, mask.sum().item())
    elif rank is not None:
        # Keep exactly 'rank' singular values (or fewer if matrix rank is lower)
        keep = min(rank, S.shape[0])
    elif threshold is not None:
        # Keep all singular values above threshold
        mask = S > threshold
        keep = mask.sum().item()
    else:
        # Keep all
        keep = S.shape[0]
    
    if keep < S.shape[0]:
        U = U[:, :keep]
        S = S[:keep]
        Vh = Vh[:keep, :]
    
    return U, S, Vh


def rl_orthogonal_svd(
    tt_cores,
    R,
    instr,
    target_ranks=None,
    threshold=None,
    zero_truncate=False,
    return_info=False,
):
    """
    Orthogonalize the TT-cores right to left using SVD decomposition.
    
    Parameters
    ----------
    tt_cores : list[torch.Tensor]
        TT cores ordered left->right.
    R : list[int]
        Original TT ranks including boundary 1s.
    instr : list
        Equivariant instructions for each core.
    target_ranks : list[int], optional
        Desired (effective) ranks for each internal bond.
    threshold : float, optional
        Singular-value threshold. When zero_truncate=True this is interpreted
        as a ratio with respect to the maximum singular value (σ / σ_max).
    zero_truncate : bool, default False
        When True, keep the tensor core shapes unchanged but zero out the
        smallest singular values instead of truncating dimensions.
    return_info : bool, default False
        When True, also return bookkeeping information (e.g. kept singular
        counts per bond).
    
    Returns
    -------
    cores_new : list[torch.Tensor]
        Updated TT cores.
    R_new : list[int]
        Updated TT ranks (equal to original ranks when zero_truncate=True).
    info : dict (optional)
        Contains metadata such as ``kept_counts`` when return_info=True.
    """
    
    d = len(tt_cores)
    
    # Init instr - EXACTLY like QR  
    cores_new = d*[None]
    cores_new[-1] = tt_cores[-1] + 0
    
    kept_counts = [None] * (d - 1)

    # Process right to left - EXACTLY like QR
    for i in range(d-1, 0, -1):
        # Init instr - EXACTLY like QR
        lmax = max([el[0] for el in instr[i]])
        ind_left = [[ii for ii, ir in enumerate(instr[i-1]) if ir[-1] == ll] for ll in range(lmax+1)]
        ind_right = [[ii for ii, ir in enumerate(instr[i]) if ir[0] == ll] for ll in range(lmax+1)]
        
        # Determine target/effective rank
        if target_ranks is not None:
            target_rank = target_ranks[i - 1] if i - 1 < len(target_ranks) else None
        else:
            target_rank = None
        preserve_shape = zero_truncate

        if preserve_shape:
            max_rnew = cores_new[i].shape[1]
            cores_new_updated = tn.zeros_like(cores_new[i])
        else:
            # First pass: determine the maximum new rank across all l values
            max_rnew = 0
            for l in range(lmax + 1):
                mode_shape = [cores_new[i].shape[2]]
                core_now = (
                    tn.stack([cores_new[i][ind, ...] for ind in ind_right[l]], dim=-3).flatten(1)
                ).t()
                U, _, _ = SVD(core_now, rank=target_rank, threshold=threshold)
                rnew = U.shape[1]
                max_rnew = max(max_rnew, rnew)

            # Allocate new core with the maximum rank needed
            if max_rnew != cores_new[i].shape[1]:
                old_shape = cores_new[i].shape
                cores_new_updated = tn.zeros(
                    [old_shape[0], max_rnew, old_shape[2], old_shape[3]],
                    device=tt_cores[0].device,
                    dtype=tt_cores[0].dtype,
                )
            else:
                cores_new_updated = cores_new[i]
        
        core_next = tt_cores[i - 1]
        
        # Second pass: actually fill in the new core
        for l in range(lmax + 1):
            mode_shape = [cores_new[i].shape[2]]
            core_now = (
                tn.stack([cores_new[i][ind, ...] for ind in ind_right[l]], dim=-3).flatten(1)
            ).t()

            rank_orig = cores_new[i].shape[1]

            if preserve_shape:
                U, S_full, Vh = tn.linalg.svd(core_now, full_matrices=False)
                S = S_full.clone()
                if S.numel() > 0:
                    sigma_max = S.max()
                    if threshold is not None and sigma_max > 0:
                        ratios = S / sigma_max
                        S[ratios < threshold] = 0
                    if target_rank is not None and target_rank < S.shape[0]:
                        S[target_rank:] = 0
            else:
                U, S, Vh = SVD(core_now, rank=target_rank, threshold=threshold)

            rnew = U.shape[1]
            kept = int((S != 0).sum().item()) if S.numel() > 0 else 0
            bond_index = i - 1
            if kept_counts[bond_index] is None or kept > kept_counts[bond_index]:
                kept_counts[bond_index] = kept

            # update current core - U plays role of Q
            cores_new_tmp = tn.reshape(U.T, [rnew] + [len(ind_right[l])] + mode_shape + [-1])

            if preserve_shape:
                cores_new_updated[ind_right[l]] = cores_new_tmp.transpose(0, 1)
            else:
                cores_new_updated[ind_right[l]] = cores_new_tmp.transpose(0, 1)

            R[i] = cores_new_updated.shape[1]

            # and the i-1 one - (S @ Vh).T plays role of R.T
            mode_shape = [core_next.shape[2]]
    
            core_next_tmp = tn.reshape(core_next[ind_left[l]],[len(ind_left[l])*core_next.shape[1]*core_next.shape[2],-1])
            # For SVD: M = U @ diag(S) @ Vh
            # We need: (diag(S) @ Vh).T = Vh.T @ diag(S)
            # Vh.T has shape [num_cols, rnew], S has shape [rnew]
            # Multiply each column of Vh.T by corresponding S
            SVh_T = Vh.T * S[None, :]  # Broadcasting: S[None, :] creates [1, rnew], broadcasts to [num_cols, rnew]
            core_next_tmp = core_next_tmp @ SVh_T
            
            if l == 0:
                cores_new[i - 1] = tn.zeros(
                    [len(instr[i - 1])] + [core_next.shape[1]] + mode_shape + [cores_new_updated.shape[1]],
                    device=core_next.device,
                    dtype=core_next.dtype,
                )
            
            cores_new[i - 1][ind_left[l]] = tn.reshape(core_next_tmp, [len(ind_left[l])] + [core_next.shape[1]] + mode_shape + [-1])
        
        # Update the core with the new rank
        cores_new[i] = cores_new_updated
        if kept_counts[i - 1] is None:
            kept_counts[i - 1] = cores_new_updated.shape[1]
        
    if return_info:
        info = {"kept_counts": kept_counts}
        return cores_new, R, info
    return cores_new, R


def rl_orthogonal_qr(tt_cores, R, instr):
    """
    Original QR-based orthogonalization (for comparison/testing).
    This is essentially the same as the original rl_orthogonal.
    """
    
    d = len(tt_cores)
    cores_new = [None] * d
    cores_new[-1] = tt_cores[-1] + 0
    
    for i in range(d-1, 0, -1):
        # Init instr
        lmax = max([el[0] for el in instr[i]])
        ind_left = [[ii for ii, ir in enumerate(instr[i-1]) if ir[-1] == ll] 
                    for ll in range(lmax+1)]
        ind_right = [[ii for ii, ir in enumerate(instr[i]) if ir[0] == ll] 
                     for ll in range(lmax+1)]
        
        core_next = tt_cores[i - 1]
        for l in range(lmax+1):
            mode_shape = [cores_new[i].shape[2]]
            core_now = (tn.stack([cores_new[i][ind, ...] for ind in ind_right[l]], 
                                 dim=-3).flatten(1)).t()
            
            # QR decomposition
            Qmat, Rmat = tn.linalg.qr(core_now)
            rnew = Rmat.shape[0]
            
            # update current core
            cores_new_tmp = tn.reshape(Qmat.T, [rnew] + [len(ind_right[l])] + 
                                      mode_shape + [-1])
            cores_new[i][ind_right[l]] = cores_new_tmp.transpose(0, 1)
            
            R[i] = cores_new[i].shape[1]
            
            # and the i-1 one
            mode_shape = [core_next.shape[2]]
            core_next_tmp = tn.reshape(core_next[ind_left[l]],
                                      [len(ind_left[l]) * core_next.shape[1] * 
                                       core_next.shape[2], -1])
            core_next_tmp = core_next_tmp @ Rmat.T
            
            if l == 0:
                cores_new[i - 1] = tn.zeros([len(instr[i-1])] + [core_next.shape[1]] + 
                                            mode_shape + [cores_new[i].shape[1]], 
                                            device=tt_cores[0].device)
            
            cores_new[i - 1][ind_left[l]] = tn.reshape(core_next_tmp, 
                                                        [len(ind_left[l])] + 
                                                        [core_next.shape[1]] + 
                                                        mode_shape + [-1])
        
    return cores_new, R


if __name__ == "__main__":
    print("SVD-based rank reduction for equivariant TT-cores")
    print("This module provides rl_orthogonal_svd for rank reduction.")
    print()
    print("Usage:")
    print("  from rank_reduction_svd import rl_orthogonal_svd")
    print()
    print("  # Full rank (same as QR):")
    print("  cores_new, R_new = rl_orthogonal_svd(cores, R, instructions)")
    print()
    print("  # With rank reduction:")
    print("  cores_new, R_new = rl_orthogonal_svd(cores, R, instructions,")
    print("                                        target_ranks=[8, 8, 8])")

