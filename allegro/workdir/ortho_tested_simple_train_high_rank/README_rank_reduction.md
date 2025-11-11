# Rank Reduction Evaluation - Updated Implementation

## Changes Made

The evaluation script has been updated to include a **baseline evaluation** before applying any orthogonalization. This allows us to properly compare:

1. **Baseline**: Model performance with the original trained weights (no orthogonalization)
2. **After Orthogonalization**: Model performance after applying `rl_orthogonal` with different target ranks

## Key Improvements

### 1. Baseline Evaluation
- The script now first loads the model and evaluates it WITHOUT any modifications
- This provides a true reference point for comparison
- Original ranks are extracted and displayed
- Both train and validation metrics are computed

### 2. Clearer Labels
- Results are now labeled as:
  - `Baseline [16, 16, 16]` - Original model before any changes
  - `After ortho [16, 16, 16]` - Model after applying orthogonalization with target rank [16, 16, 16]
  - etc.

### 3. Enhanced Visualization
- The analysis notebook now shows:
  - Baseline metrics as horizontal dashed lines (reference)
  - Post-orthogonalization metrics as points/lines
  - Easy visual comparison between baseline and orthogonalized results

## What to Look For in Results

### If Orthogonalization Works:
1. **No Rank Change**: If all "After ortho" results show the same rank as baseline:
   - `rl_orthogonal` is performing orthogonalization but NOT reducing ranks
   - QR decomposition preserves the full rank
   - Slight accuracy changes may indicate numerical effects of orthogonalization

2. **Rank Changes**: If different ranks appear:
   - Ranks successfully reduced to target values
   - Can analyze accuracy degradation vs compression ratio
   - Can identify optimal rank for deployment

### Metrics to Compare:
- **Force MAE/RMSE**: Primary metric for atomic force prediction accuracy
- **Energy MAE**: Total energy prediction accuracy
- **Energy/Atom MAE**: Per-atom energy accuracy (normalized)
- **Parameter Count**: Model size (should decrease with rank reduction)
- **Compression Ratio**: Original params / reduced params

## Running the Evaluation

```bash
cd /home/vladimir/DATA/linux_data/GitHub/SpinGNN/allegro/workdir/ortho_tested_simple_train_high_rank
source /home/vladimir/DATA/linux_data/GitHub/spingnn/bin/activate
python test_rank_reduction.py
```

This will generate:
- `rank_reduction_results.csv` - Detailed metrics for all configurations
- Console output with progress and summary

## Analyzing Results

Open and run the Jupyter notebook:
```bash
jupyter notebook rank_reduction_analysis.ipynb
```

The notebook will:
1. Load and parse the CSV results
2. Separate baseline from orthogonalized results
3. Create comparison plots showing:
   - Baseline performance (horizontal lines)
   - Post-orthogonalization performance (points/curves)
4. Generate detailed comparison tables
5. Provide analysis and observations

## Next Steps

Based on the results, you can:

1. **If ranks don't reduce**: 
   - Investigate `rl_orthogonal` implementation
   - Consider SVD-based rank truncation
   - Examine singular values of TT-cores

2. **If ranks reduce successfully**:
   - Identify optimal rank for your accuracy requirements
   - Analyze the trade-off between compression and accuracy
   - Deploy the compressed model if acceptable

3. **If orthogonalization affects accuracy**:
   - Understand if it's beneficial (regularization effect)
   - Or detrimental (numerical instability)
   - Decide if orthogonalization should be applied

