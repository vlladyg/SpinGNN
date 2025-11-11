# SVD-Based Rank Reduction: Implementation Summary

## Status: ✓ SVD Implemented and Validated

### What Works

✓ **SVD-based orthogonalization with full rank**
- Implementation in `rank_reduction_svd.py`
- Validated against QR implementation
- Produces identical predictions (differences < 1e-06)
- Test: `test_svd_vs_qr_predictions.py` ✓ PASSED

### Key Finding: Rank Reduction Requires Model Rebuild

**Important Discovery**: Simply swapping TT-cores with reduced ranks does not work for inference because:

1. **TorchScript Compilation**: The model uses JIT-compiled forward passes with hardcoded dimensions
2. **TensorProduct Layers**: Intermediate layers (e.g., `tps`) have baked-in dimension expectations
3. **Architecture Dependency**: The ETN model's architecture is tightly coupled to the rank dimensions

#### Error Example
```
RuntimeError: shape '[11, 16, 16, 16]' is invalid for input of size 34496
```
When ranks are reduced from [16,16,16] to [14,14,14], the TensorProduct layers still expect rank-16 tensors.

### Two Approaches for Rank Reduction

#### Approach 1: Rank Reduction via Retraining (Recommended)

1. **Initialize with reduced rank**: Train a new model with target_rank=[8,8,8] from scratch
2. **Transfer knowledge**: Optionally initialize from compressed cores of full-rank model
3. **Fine-tune**: Train for full epochs to learn optimal compressed representation

**Advantages**:
- Model architecture is consistent
- Can achieve better accuracy through optimization
- Clean, supported approach

#### Approach 2: Direct Core Compression (Not Feasible Currently)

1. Apply SVD to reduce TT-core ranks
2. **Problem**: Would need to rebuild entire model architecture to match new ranks
3. **Complexity**: Requires modifying TensorProduct layers, buffers, and JIT-compiled code

**Conclusion**: Not practical without significant architectural changes.

## What We've Validated

### SVD Implementation

✓ **Correctness**: SVD produces identical predictions to QR at full rank

| Metric | QR | SVD | Difference |
|--------|-----|-----|------------|
| Train F MAE | 0.04093250 | 0.04093246 | 3.5e-08 |
| Val F MAE | 0.03500310 | 0.03500304 | 5.6e-08 |
| Train E/N MAE | 0.00526150 | 0.00526087 | 6.4e-07 |
| Val E/N MAE | 0.00486748 | 0.00486687 | 6.1e-07 |

✓ **Tensor Decomposition**: SVD correctly decomposes TT-cores while preserving equivariant structure

✓ **Rank Truncation Logic**: The SVD function can truncate ranks (we tested this on isolated cores)

### Implementation Files

Created:
- `rank_reduction_svd.py`: Core SVD implementation
- `test_svd_vs_qr_predictions.py`: Validation test (PASSED)
- `README_SVD_rank_reduction.md`: Documentation
- `debug_qr_vs_svd.py`: Mathematical verification

## Recommendations

### For Model Compression

**Use Approach 1 (Retraining with Reduced Rank)**:

1. **Modify training config**:
   ```yaml
   # In config/example_ETN_opt_MEA.yaml
   N_ranks_ett: [8, 8, 8]  # Instead of [16, 16, 16]
   ```

2. **Train from scratch** or **initialize from compressed cores**:
   ```python
   # Load full-rank model
   full_model = load_model("best_model.pth")
   
   # Apply SVD compression to get initial cores
   cores_compressed, _ = rl_orthogonal_svd(
       full_model.cores, 
       ranks, 
       instructions,
       target_ranks=[8, 8, 8]
   )
   
   # Build new model with rank-8 architecture
   config['N_ranks_ett'] = [8, 8, 8]
   compressed_model = model_from_config(config)
   
   # Initialize with compressed cores
   compressed_model.cores = cores_compressed
   
   # Fine-tune
   train(compressed_model, ...)
   ```

3. **Evaluate compression vs accuracy trade-off**:
   - Test ranks: [12,12,12], [8,8,8], [6,6,6], [4,4,4]
   - Measure: parameter count, accuracy, training time

### For Future Development

If direct core-swapping is desired, would need to:
1. Disable JIT compilation (`@torch.jit.script` decorators)
2. Make TensorProduct layers dynamically adapt to input ranks
3. Update all hardcoded dimension checks
4. Rebuild model buffers and intermediate layers

**Estimated effort**: Significant architectural refactoring

## Conclusion

✓ **SVD implementation is correct** and ready to use for:
- Full-rank orthogonalization (numerical stability)
- Generating initial cores for rank-reduced models

⚠ **Rank reduction for inference requires**:
- Retraining with target rank (recommended)
- Or significant architectural changes (not recommended)

The SVD-based compression can be valuable as:
1. **Initialization method**: Compress cores from full-rank model, then retrain
2. **Analysis tool**: Understand which ranks are most important (via singular values)
3. **Research tool**: Study rank-accuracy trade-offs during training

## Files and Tests

All implementation files are in:
```
allegro/workdir/ortho_tested_simple_train_high_rank/
├── rank_reduction_svd.py              # ✓ SVD implementation
├── test_svd_vs_qr_predictions.py      # ✓ Validation (PASSED)
├── README_SVD_rank_reduction.md       # Documentation
├── debug_qr_vs_svd.py                 # Math verification
└── THIS_FILE.md                       # Summary
```

## Next Steps

1. **Test retraining approach**: Train models with reduced ranks from scratch
2. **Benchmarking**: Compare training time and accuracy for different ranks
3. **Transfer learning**: Test initializing from SVD-compressed cores
4. **Integration**: Add to main codebase once approach is finalized

---

**Bottom Line**: The SVD implementation works correctly. For actual model compression, retrain with the target rank rather than swapping cores post-training.

