"""
Test script to compare QR-based and SVD-based orthogonalization.

When using full rank (no truncation), SVD should produce equivalent results to QR.
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nequip.train import Trainer
from nequip.data import AtomicDataDict, dataset_from_config
from nequip.utils import Config
from nequip.model import model_from_config
from rank_reduction_svd import rl_orthogonal_svd, rl_orthogonal_qr
from allegro.utils import rl_orthogonal

# Default config for loading
default_config = dict(
    root="./",
    tensorboard=False,
    wandb=False,
    model_builders=[
        "SimpleIrrepsConfig",
        "EnergyModel",
        "PerSpeciesRescale",
        "StressForceOutput",
        "RescaleEnergyEtc",
    ],
    dataset_statistics_stride=1,
    device='cuda:0',
    default_dtype="float64",
    model_dtype="float32",
    allow_tf32=True,
    verbose="INFO",
    model_debug_mode=False,
    equivariance_test=False,
    grad_anomaly_mode=False,
    gpu_oom_offload=False,
    append=False,
    warn_unused=False,
    _jit_bailout_depth=2,
    _jit_fusion_strategy=[("DYNAMIC", 3)],
    _jit_fuser="fuser1",
)

os.environ['NEQUIP_NUM_TASKS'] = '1'


def compute_reconstruction_error(cores1, cores2):
    """Compute the relative error between two sets of TT-cores."""
    total_error = 0.0
    total_norm = 0.0
    
    for c1, c2 in zip(cores1, cores2):
        error = torch.norm(c1 - c2).item()
        norm = torch.norm(c1).item()
        total_error += error ** 2
        total_norm += norm ** 2
    
    return np.sqrt(total_error / total_norm)


def test_orthogonalization_equivalence(config_path="results/MEA_Allegro_0/example/config.yaml",
                                      model_dir="results/MEA_Allegro_0/example", 
                                      model_name="best_model.pth", device='cpu'):
    """
    Test that SVD-based orthogonalization with full rank matches QR-based version.
    """
    
    print("="*80)
    print("TESTING: SVD vs QR Orthogonalization (Full Rank)")
    print("="*80)
    print()
    
    # Load config
    print("Loading configuration...")
    config = Config.from_file(config_path, defaults=default_config)
    device = torch.device(config.get('device', device))
    print(f"  Using device: {device}")
    
    # Load dataset (needed for model initialization)
    print("\nLoading dataset...")
    config['root'] = './results/MEA_Allegro_0'
    full_dataset = dataset_from_config(config, prefix="dataset")
    print(f"  Dataset size: {len(full_dataset)}")
    
    # Load model
    print("\nLoading model...")
    model = model_from_config(
        config=config, 
        initialize=True,
        dataset=full_dataset
    )
    state_dict = torch.load(f"{model_dir}/{model_name}", map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # Get TT-cores and structure info
    cores_original = model.get_submodule('model.model.func.etn.cores')
    instructions = []
    for i in range(config['d']):
        instructions.append([tuple(el) for el in 
                           model.get_buffer(f'model.model.func.etn.instructions_list_{i}').tolist()])
    
    original_ranks = model.get_buffer(f'model.model.func.etn.N_rank_ett').tolist()
    ranks = [1] + original_ranks + [1]
    
    print(f"Model configuration:")
    print(f"  d = {config['d']}")
    print(f"  Original ranks: {original_ranks}")
    print()
    
    # Make copies of cores for each test
    cores_qr_orig = [c.clone() for c in cores_original]
    cores_qr_new = [c.clone() for c in cores_original]
    cores_svd_new = [c.clone() for c in cores_original]
    
    # Test 1: Original QR implementation
    print("Test 1: Original QR implementation (rl_orthogonal)")
    print("-" * 80)
    ranks_qr_orig = ranks.copy()
    cores_qr_orig, ranks_qr_orig = rl_orthogonal(cores_qr_orig, ranks_qr_orig, instructions)
    print(f"  Output ranks: {ranks_qr_orig[1:-1]}")
    print()
    
    # Test 2: New QR implementation (for verification)
    print("Test 2: New QR implementation (rl_orthogonal_qr)")
    print("-" * 80)
    ranks_qr_new = ranks.copy()
    cores_qr_new, ranks_qr_new = rl_orthogonal_qr(cores_qr_new, ranks_qr_new, instructions)
    print(f"  Output ranks: {ranks_qr_new[1:-1]}")
    
    # Compare new QR with original QR
    error_qr = compute_reconstruction_error(cores_qr_orig, cores_qr_new)
    print(f"  Relative error vs original QR: {error_qr:.2e}")
    print()
    
    # Test 3: SVD with full rank (should match QR)
    print("Test 3: SVD implementation with full rank (rl_orthogonal_svd)")
    print("-" * 80)
    ranks_svd = ranks.copy()
    # Pass None for target_ranks to keep full rank
    cores_svd_new, ranks_svd = rl_orthogonal_svd(cores_svd_new, ranks_svd, instructions, 
                                                   target_ranks=None, threshold=None)
    print(f"  Output ranks: {ranks_svd[1:-1]}")
    
    # Compare SVD with QR
    error_svd_vs_qr = compute_reconstruction_error(cores_qr_orig, cores_svd_new)
    print(f"  Relative error vs original QR: {error_svd_vs_qr:.2e}")
    print()
    
    # Comparison summary
    print("="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"Original QR ranks:  {ranks_qr_orig[1:-1]}")
    print(f"New QR ranks:       {ranks_qr_new[1:-1]}")
    print(f"SVD ranks:          {ranks_svd[1:-1]}")
    print()
    print(f"New QR vs Original QR error:  {error_qr:.2e}")
    print(f"SVD vs Original QR error:     {error_svd_vs_qr:.2e}")
    print()
    
    # Determine pass/fail
    tolerance = 1e-5
    if error_qr < tolerance and error_svd_vs_qr < tolerance:
        print("✓ PASS: SVD with full rank matches QR implementation")
        print(f"  Both errors < {tolerance}")
    else:
        print("✗ FAIL: Implementations do not match")
        if error_qr >= tolerance:
            print(f"  New QR implementation differs from original (error: {error_qr:.2e})")
        if error_svd_vs_qr >= tolerance:
            print(f"  SVD implementation differs from QR (error: {error_svd_vs_qr:.2e})")
    
    print("="*80)
    print()
    
    return error_svd_vs_qr < tolerance


def test_svd_rank_reduction(config_path="results/MEA_Allegro_0/example/config.yaml",
                            model_dir="results/MEA_Allegro_0/example", 
                            model_name="best_model.pth", 
                            target_ranks=[8, 8, 8], device='cpu'):
    """
    Test SVD-based rank reduction with truncation.
    """
    
    print("="*80)
    print(f"TESTING: SVD Rank Reduction to {target_ranks}")
    print("="*80)
    print()
    
    # Load config
    print("Loading configuration...")
    config = Config.from_file(config_path, defaults=default_config)
    device = torch.device(config.get('device', device))
    
    # Load dataset
    print("\nLoading dataset...")
    config['root'] = './results/MEA_Allegro_0'
    full_dataset = dataset_from_config(config, prefix="dataset")
    
    # Load model
    print("\nLoading model...")
    model = model_from_config(
        config=config, 
        initialize=True,
        dataset=full_dataset
    )
    state_dict = torch.load(f"{model_dir}/{model_name}", map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # Get TT-cores and structure info
    cores_original = [c.clone() for c in model.get_submodule('model.model.func.etn.cores')]
    cores_reduced = [c.clone() for c in cores_original]
    
    instructions = []
    for i in range(config['d']):
        instructions.append([tuple(el) for el in 
                           model.get_buffer(f'model.model.func.etn.instructions_list_{i}').tolist()])
    
    original_ranks = model.get_buffer(f'model.model.func.etn.N_rank_ett').tolist()
    ranks = [1] + original_ranks + [1]
    
    print(f"Model configuration:")
    print(f"  Original ranks: {original_ranks}")
    print(f"  Target ranks: {target_ranks}")
    print()
    
    # Apply SVD with rank reduction
    print("Applying SVD rank reduction...")
    ranks_reduced = ranks.copy()
    cores_reduced, ranks_reduced = rl_orthogonal_svd(cores_reduced, ranks_reduced, 
                                                       instructions, target_ranks=target_ranks)
    
    print(f"  Resulting ranks: {ranks_reduced[1:-1]}")
    print()
    
    # Compute approximation error
    error = compute_reconstruction_error(cores_original, cores_reduced)
    print(f"Reconstruction error: {error:.2e}")
    print()
    
    # Count parameters
    def count_params(cores):
        return sum(c.numel() for c in cores)
    
    n_params_orig = count_params(cores_original)
    n_params_reduced = count_params(cores_reduced)
    compression = n_params_orig / n_params_reduced
    
    print(f"Parameters:")
    print(f"  Original:  {n_params_orig:,}")
    print(f"  Reduced:   {n_params_reduced:,}")
    print(f"  Compression: {compression:.2f}x")
    print()
    
    print("="*80)
    print()
    
    return ranks_reduced[1:-1], error, compression


if __name__ == "__main__":
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*20 + "SVD vs QR ORTHOGONALIZATION TEST" + " "*26 + "║")
    print("╚" + "="*78 + "╝")
    print()
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print()
    
    # Test 1: Equivalence at full rank
    print("\n" + "╔" + "="*78 + "╗")
    print("║" + " "*25 + "TEST 1: Full Rank Equivalence" + " "*24 + "║")
    print("╚" + "="*78 + "╝")
    print()
    
    try:
        passed = test_orthogonalization_equivalence(device=device)
        
        if passed:
            # Test 2: Rank reduction (only if equivalence test passed)
            print("\n" + "╔" + "="*78 + "╗")
            print("║" + " "*27 + "TEST 2: Rank Reduction" + " "*28 + "║")
            print("╚" + "="*78 + "╝")
            print()
            
            # Test different ranks
            for target_ranks in [[16, 16, 16], [12, 12, 12], [8, 8, 8]]:
                test_svd_rank_reduction(target_ranks=target_ranks, device=device)
        else:
            print("\n⚠ Skipping rank reduction tests due to equivalence test failure")
    
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "╔" + "="*78 + "╗")
    print("║" + " "*32 + "TESTS COMPLETE" + " "*32 + "║")
    print("╚" + "="*78 + "╝")
    print()

