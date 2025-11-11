"""
Test script to compare QR-based and SVD-based orthogonalization.

When using full rank (no truncation), SVD should produce the same model predictions as QR.
We compare energy and force errors on train/val datasets.
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nequip.train import Trainer
from nequip.data import AtomicDataDict, dataset_from_config, DataLoader, Collater
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


def compute_metrics(model, data_loader, device):
    """
    Compute metrics for a model on a dataset.
    
    Returns:
        dict with keys: f_mae, f_rmse, e_mae, e_per_atom_mae
    """
    model.eval()
    
    total_f_error = 0.0
    total_f_sq_error = 0.0
    total_e_error = 0.0
    total_e_per_atom_error = 0.0
    total_force_count = 0
    total_energy_count = 0
    
    for batch_idx, batch in enumerate(data_loader):
        batch = batch.to(device)
        batch_dict = {key: batch[key] for key in batch.keys}
        
        if AtomicDataDict.POSITIONS_KEY in batch_dict:
            batch_dict[AtomicDataDict.POSITIONS_KEY] = batch_dict[AtomicDataDict.POSITIONS_KEY].requires_grad_(True)
        
        pred = model(batch_dict)
        
        # Forces
        if AtomicDataDict.FORCE_KEY in batch_dict:
            forces_true = batch_dict[AtomicDataDict.FORCE_KEY]
            forces_pred = pred[AtomicDataDict.FORCE_KEY]
            
            f_error = torch.abs(forces_pred - forces_true)
            total_f_error += f_error.sum().item()
            total_f_sq_error += (f_error ** 2).sum().item()
            total_force_count += forces_true.numel()
        
        # Energy
        if AtomicDataDict.TOTAL_ENERGY_KEY in batch_dict:
            energy_true = batch_dict[AtomicDataDict.TOTAL_ENERGY_KEY]
            energy_pred = pred[AtomicDataDict.TOTAL_ENERGY_KEY]
            
            e_error = torch.abs(energy_pred - energy_true)
            total_e_error += e_error.sum().item()
            total_energy_count += energy_true.numel()
            
            n_atoms = torch.bincount(batch_dict[AtomicDataDict.BATCH_KEY]).float()
            e_per_atom_error = torch.abs((energy_pred - energy_true) / n_atoms.unsqueeze(-1))
            total_e_per_atom_error += e_per_atom_error.sum().item()
        
        if AtomicDataDict.POSITIONS_KEY in batch_dict and batch_dict[AtomicDataDict.POSITIONS_KEY].grad is not None:
            batch_dict[AtomicDataDict.POSITIONS_KEY].grad.zero_()
    
    return {
        'f_mae': total_f_error / total_force_count if total_force_count > 0 else 0,
        'f_rmse': np.sqrt(total_f_sq_error / total_force_count) if total_force_count > 0 else 0,
        'e_mae': total_e_error / total_energy_count if total_energy_count > 0 else 0,
        'e_per_atom_mae': total_e_per_atom_error / total_energy_count if total_energy_count > 0 else 0,
    }


def test_orthogonalization_equivalence(config_path="results/MEA_Allegro_0/example/config.yaml",
                                      model_dir="results/MEA_Allegro_0/example", 
                                      model_name="best_model.pth", device='cpu'):
    """
    Test that SVD-based orthogonalization with full rank produces the same predictions as QR.
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
    
    # Load dataset
    print("\nLoading dataset...")
    config['root'] = './results/MEA_Allegro_0'
    full_dataset = dataset_from_config(config, prefix="dataset")
    print(f"  Dataset size: {len(full_dataset)}")
    
    # Create train/val split
    n_train = config['n_train']
    n_val = config['n_val']
    
    if config['train_val_split'] == 'random':
        import random
        random.seed(config.get('dataset_seed', 123456))
        indices = list(range(len(full_dataset)))
        random.shuffle(indices)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
    else:
        train_indices = list(range(n_train))
        val_indices = list(range(n_train, n_train + n_val))
    
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    
    batch_size = config.get('batch_size', 10)
    collater = Collater.for_dataset(full_dataset)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collater)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collater)
    
    print(f"  Train samples: {n_train}, Val samples: {n_val}")
    
    # Load original model
    print("\nLoading original model...")
    model_orig = model_from_config(config=config, initialize=True, dataset=full_dataset)
    state_dict = torch.load(f"{model_dir}/{model_name}", map_location=device)
    model_orig.load_state_dict(state_dict)
    model_orig.to(device)
    model_orig.eval()
    
    for param in model_orig.parameters():
        param.requires_grad = False
    
    original_ranks = model_orig.get_buffer(f'model.model.func.etn.N_rank_ett').tolist()
    print(f"  Original ranks: {original_ranks}")
    
    # Evaluate original model
    print("\nEvaluating original model (before orthogonalization)...")
    orig_train_metrics = compute_metrics(model_orig, train_loader, device)
    orig_val_metrics = compute_metrics(model_orig, val_loader, device)
    print(f"  Train - F MAE: {orig_train_metrics['f_mae']:.6f}, E/N MAE: {orig_train_metrics['e_per_atom_mae']:.6f}")
    print(f"  Val   - F MAE: {orig_val_metrics['f_mae']:.6f}, E/N MAE: {orig_val_metrics['e_per_atom_mae']:.6f}")
    
    # Test QR orthogonalization
    print("\n" + "-"*80)
    print("Test 1: QR Orthogonalization (rl_orthogonal)")
    print("-"*80)
    
    model_qr = model_from_config(config=config, initialize=True, dataset=full_dataset)
    model_qr.load_state_dict(state_dict)
    model_qr.to(device)
    model_qr.eval()
    
    for param in model_qr.parameters():
        param.requires_grad = False
    
    # Apply QR orthogonalization
    cores_qr = model_qr.get_submodule('model.model.func.etn.cores')
    instructions = []
    for i in range(config['d']):
        instructions.append([tuple(el) for el in 
                           model_qr.get_buffer(f'model.model.func.etn.instructions_list_{i}').tolist()])
    
    ranks_qr = [1] + original_ranks + [1]
    cores_qr_new, ranks_qr_new = rl_orthogonal(cores_qr, ranks_qr.copy(), instructions)
    
    for i in range(config['d']):
        model_qr.get_submodule('model.model.func.etn.cores')[i] = cores_qr_new[i]
    
    print(f"  Ranks after QR: {ranks_qr_new[1:-1]}")
    
    # Evaluate QR model
    print("  Evaluating QR orthogonalized model...")
    qr_train_metrics = compute_metrics(model_qr, train_loader, device)
    qr_val_metrics = compute_metrics(model_qr, val_loader, device)
    print(f"  Train - F MAE: {qr_train_metrics['f_mae']:.6f}, E/N MAE: {qr_train_metrics['e_per_atom_mae']:.6f}")
    print(f"  Val   - F MAE: {qr_val_metrics['f_mae']:.6f}, E/N MAE: {qr_val_metrics['e_per_atom_mae']:.6f}")
    
    # Test SVD orthogonalization
    print("\n" + "-"*80)
    print("Test 2: SVD Orthogonalization (rl_orthogonal_svd, full rank)")
    print("-"*80)
    
    model_svd = model_from_config(config=config, initialize=True, dataset=full_dataset)
    model_svd.load_state_dict(state_dict)
    model_svd.to(device)
    model_svd.eval()
    
    for param in model_svd.parameters():
        param.requires_grad = False
    
    # Apply SVD orthogonalization (no rank reduction)
    cores_svd = model_svd.get_submodule('model.model.func.etn.cores')
    ranks_svd = [1] + original_ranks + [1]
    cores_svd_new, ranks_svd_new = rl_orthogonal_svd(cores_svd, ranks_svd.copy(), instructions, 
                                                       target_ranks=None, threshold=None)
    
    for i in range(config['d']):
        model_svd.get_submodule('model.model.func.etn.cores')[i] = cores_svd_new[i]
    
    print(f"  Ranks after SVD: {ranks_svd_new[1:-1]}")
    
    # Evaluate SVD model
    print("  Evaluating SVD orthogonalized model...")
    svd_train_metrics = compute_metrics(model_svd, train_loader, device)
    svd_val_metrics = compute_metrics(model_svd, val_loader, device)
    print(f"  Train - F MAE: {svd_train_metrics['f_mae']:.6f}, E/N MAE: {svd_train_metrics['e_per_atom_mae']:.6f}")
    print(f"  Val   - F MAE: {svd_val_metrics['f_mae']:.6f}, E/N MAE: {svd_val_metrics['e_per_atom_mae']:.6f}")
    
    # Comparison
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    
    print("\nTrain Set:")
    print(f"  Original - F MAE: {orig_train_metrics['f_mae']:.8f}, E/N MAE: {orig_train_metrics['e_per_atom_mae']:.8f}")
    print(f"  QR       - F MAE: {qr_train_metrics['f_mae']:.8f}, E/N MAE: {qr_train_metrics['e_per_atom_mae']:.8f}")
    print(f"  SVD      - F MAE: {svd_train_metrics['f_mae']:.8f}, E/N MAE: {svd_train_metrics['e_per_atom_mae']:.8f}")
    print(f"\n  QR vs SVD difference:")
    print(f"    F MAE diff: {abs(qr_train_metrics['f_mae'] - svd_train_metrics['f_mae']):.2e}")
    print(f"    E/N MAE diff: {abs(qr_train_metrics['e_per_atom_mae'] - svd_train_metrics['e_per_atom_mae']):.2e}")
    
    print("\nValidation Set:")
    print(f"  Original - F MAE: {orig_val_metrics['f_mae']:.8f}, E/N MAE: {orig_val_metrics['e_per_atom_mae']:.8f}")
    print(f"  QR       - F MAE: {qr_val_metrics['f_mae']:.8f}, E/N MAE: {qr_val_metrics['e_per_atom_mae']:.8f}")
    print(f"  SVD      - F MAE: {svd_val_metrics['f_mae']:.8f}, E/N MAE: {svd_val_metrics['e_per_atom_mae']:.8f}")
    print(f"\n  QR vs SVD difference:")
    print(f"    F MAE diff: {abs(qr_val_metrics['f_mae'] - svd_val_metrics['f_mae']):.2e}")
    print(f"    E/N MAE diff: {abs(qr_val_metrics['e_per_atom_mae'] - svd_val_metrics['e_per_atom_mae']):.2e}")
    
    # Determine pass/fail
    tolerance = 1e-6
    train_f_match = abs(qr_train_metrics['f_mae'] - svd_train_metrics['f_mae']) < tolerance
    train_e_match = abs(qr_train_metrics['e_per_atom_mae'] - svd_train_metrics['e_per_atom_mae']) < tolerance
    val_f_match = abs(qr_val_metrics['f_mae'] - svd_val_metrics['f_mae']) < tolerance
    val_e_match = abs(qr_val_metrics['e_per_atom_mae'] - svd_val_metrics['e_per_atom_mae']) < tolerance
    
    print("\n" + "="*80)
    if train_f_match and train_e_match and val_f_match and val_e_match:
        print("✓ PASS: SVD with full rank produces identical predictions to QR")
        print(f"  All differences < {tolerance}")
        return True
    else:
        print("✗ FAIL: SVD predictions differ from QR")
        if not train_f_match:
            print(f"  Train F MAE difference too large: {abs(qr_train_metrics['f_mae'] - svd_train_metrics['f_mae']):.2e}")
        if not train_e_match:
            print(f"  Train E/N MAE difference too large: {abs(qr_train_metrics['e_per_atom_mae'] - svd_train_metrics['e_per_atom_mae']):.2e}")
        if not val_f_match:
            print(f"  Val F MAE difference too large: {abs(qr_val_metrics['f_mae'] - svd_val_metrics['f_mae']):.2e}")
        if not val_e_match:
            print(f"  Val E/N MAE difference too large: {abs(qr_val_metrics['e_per_atom_mae'] - svd_val_metrics['e_per_atom_mae']):.2e}")
        return False


if __name__ == "__main__":
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*15 + "SVD vs QR ORTHOGONALIZATION - PREDICTION TEST" + " "*18 + "║")
    print("╚" + "="*78 + "╝")
    print()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print()
    
    try:
        passed = test_orthogonalization_equivalence(device=device)
        
        if passed:
            print("\n✓ Test passed! SVD implementation is correct.")
            print("  Ready to use for rank reduction experiments.")
        else:
            print("\n✗ Test failed! SVD implementation needs debugging.")
    
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "╔" + "="*78 + "╗")
    print("║" + " "*32 + "TEST COMPLETE" + " "*33 + "║")
    print("╚" + "="*78 + "╝")
    print()

