# SVD-Based Rank Reduction for TT-Cores

## Overview

This directory contains an SVD-based implementation of rank reduction for TT-cores with equivariant structure. Unlike the QR-based `rl_orthogonal` which only performs orthogonalization (preserving rank), the SVD version can actually truncate ranks for model compression.

## Files

- **`rank_reduction_svd.py`**: Core implementation
  - `rl_orthogonal_svd()`: SVD-based orthogonalization with optional rank truncation
  - `rl_orthogonal_qr()`: Reference QR implementation for comparison
  - `SVD()`: Helper function for SVD decomposition with truncation

- **`test_svd_vs_qr_predictions.py`**: Validation test
  - Compares model predictions (force & energy errors) between QR and SVD
  - Verifies that full-rank SVD produces identical results to QR
  - ✓ **Test Status: PASSED** (differences < 1e-06)

- **`debug_qr_vs_svd.py`**: Mathematical verification script
  - Demonstrates that QR and SVD produce different individual cores
  - But preserve the tensor contraction (thus identical predictions)

## Key Findings

### 1. Full-Rank Orthogonalization (No Compression)

When `target_ranks=None` and `threshold=None`, SVD produces identical model predictions to QR:

| Method | Train F MAE | Val F MAE | Train E/N MAE | Val E/N MAE |
|--------|-------------|-----------|---------------|-------------|
| Original | 0.04093250 | 0.03500312 | 0.00526149 | 0.00486749 |
| QR | 0.04093249 | 0.03500311 | 0.00526152 | 0.00486750 |
| SVD | 0.04093247 | 0.03500307 | 0.00526088 | 0.00486690 |

**Differences (QR vs SVD):**
- Force MAE: ~1e-08
- Energy/N MAE: ~6e-07

✓ **Conclusion**: SVD implementation is mathematically correct and ready for rank reduction experiments.

### 2. Why Individual Cores Differ

The individual TT-cores produced by QR and SVD are different (relative error ~1.3), but this is **expected and correct**:

- **QR decomposition**: M = Q @ R  
- **SVD decomposition**: M = U @ diag(S) @ Vh

Where Q ≠ U in general, but both preserve the matrix M when reconstructed.

In TT-decomposition:
- QR stores: Q^T (current core), updates left with R^T
- SVD stores: U^T (current core), updates left with Vh^T @ diag(S)

Both preserve the full tensor contraction, thus producing **identical model predictions**.

## Usage

### Full-Rank Orthogonalization

```python
from rank_reduction_svd import rl_orthogonal_svd

# Apply SVD orthogonalization (no rank reduction)
cores_new, ranks_new = rl_orthogonal_svd(
    tt_cores,
    ranks,
    instructions,
    target_ranks=None,  # Keep full rank
    threshold=None
)
```

### Rank Reduction

```python
# Reduce to rank [8, 8, 8]
cores_new, ranks_new = rl_orthogonal_svd(
    tt_cores,
    ranks,
    instructions,
    target_ranks=[8, 8, 8],  # Target ranks
    threshold=None  # Optional: singular value threshold
)
```

### With Threshold-Based Truncation

```python
# Truncate singular values below threshold
cores_new, ranks_new = rl_orthogonal_svd(
    tt_cores,
    ranks,
    instructions,
    target_ranks=[16, 16, 16],  # Max rank per position
    threshold=1e-5  # Drop singular values < 1e-5
)
```

## Comparison: QR vs SVD

| Feature | QR (`rl_orthogonal`) | SVD (`rl_orthogonal_svd`) |
|---------|---------------------|---------------------------|
| **Rank Reduction** | ❌ No (preserves full rank) | ✓ Yes (can truncate) |
| **Orthogonalization** | ✓ Yes | ✓ Yes |
| **Equivariant** | ✓ Yes | ✓ Yes |
| **Speed** | Faster (no SVD) | Slower (SVD computation) |
| **Predictions (full rank)** | Identical | Identical (verified) |
| **Use Case** | Numerical stability | Model compression |

## Implementation Details

### Right-to-Left Orthogonalization

The SVD version follows the same equivariant structure as QR:

1. Process cores from right to left (i = d-1 to 1)
2. For each core, handle l-values separately (equivariant paths)
3. Decompose reshaped core: M = U @ diag(S) @ Vh
4. Store U^T as orthogonalized core
5. Pass Vh^T @ diag(S) to the left core

### Rank Truncation

When `target_ranks` is specified:
- SVD computes all singular values
- Keeps only the top `target_rank` values
- Truncation happens automatically via matrix slicing
- Actual rank may be lower if matrix rank < target

### Singular Value Threshold

When `threshold` is specified:
- Drops singular values below threshold
- Can be combined with `target_ranks` (keeps min of both)
- Useful for automatic compression based on numerical precision

## Testing

### Run Validation Test

```bash
source /home/vladimir/DATA/linux_data/GitHub/spingnn/bin/activate
cd /home/vladimir/DATA/linux_data/GitHub/SpinGNN/allegro/workdir/ortho_tested_simple_train_high_rank
python test_svd_vs_qr_predictions.py
```

**Expected Output**: ✓ PASS (all differences < 1e-06)

### Run Mathematical Verification

```bash
python debug_qr_vs_svd.py
```

Shows that individual cores differ but tensor contraction is preserved.

## Next Steps

### 1. Rank Reduction Experiments

Test various target ranks to find the accuracy-compression trade-off:

```python
test_ranks = [
    [16, 16, 16],  # Full rank (baseline)
    [12, 12, 12],  # 25% reduction
    [8, 8, 8],     # 50% reduction
    [4, 4, 4],     # 75% reduction
]
```

### 2. Integration with Main Code

Once validated, add to `allegro/allegro/utils.py`:

```python
def rl_orthogonal_svd(tt_cores, R, instr, target_ranks=None, threshold=None):
    """SVD-based rank reduction for TT-cores."""
    # Implementation here
```

### 3. Retraining After Compression

After rank reduction, fine-tune the compressed model:
1. Apply SVD rank reduction
2. Load compressed cores into model
3. Train for a few epochs to recover accuracy

## Performance Considerations

### Computational Cost

- **QR**: O(m²n) for m×n matrix
- **SVD**: O(min(m,n) × m × n)

For typical TT-cores with small ranks, the overhead is acceptable.

### Memory

SVD requires temporarily storing U, S, Vh matrices. For large cores:
- Process in batches
- Use `torch.cuda.empty_cache()` between operations
- Consider CPU offloading for very large models

## References

- Original QR implementation: `allegro/allegro/utils.py::rl_orthogonal()`
- TT-decomposition: https://arxiv.org/abs/1509.06569
- Equivariant networks: https://arxiv.org/abs/2101.03164

## Status

✓ **Implementation Complete**  
✓ **Validation Passed** (predictions match QR at full rank)  
⏳ **Ready for rank reduction experiments**  
⏳ **Pending integration into main codebase**

