"""
Evaluate rank reduction effects on trained ETN model.

This script loads a trained ETN model, applies rl_orthogonal with different ranks,
and measures how train/validation accuracy changes without retraining.
"""

import sys
import os
import csv
from pathlib import Path
import torch
import numpy as np
from copy import deepcopy

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nequip.data import dataset_from_config, DataLoader, Collater
from nequip.utils import Config
from nequip.train.trainer import Trainer
from nequip.data import AtomicDataDict
from nequip.model import model_from_config
from allegro import rl_orthogonal

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


def compute_metrics(model, data_loader, device, loss_fn=None):
    """
    Compute metrics for a model on a dataset.
    
    Returns:
        dict with keys: f_mae, f_rmse, e_mae, e_per_atom_mae, loss
    """
    model.eval()
    
    # Accumulators
    total_f_error = 0.0
    total_f_sq_error = 0.0
    total_e_error = 0.0
    total_e_per_atom_error = 0.0
    total_loss = 0.0
    total_force_count = 0
    total_energy_count = 0
    
    
    for batch_idx, batch in enumerate(data_loader):
        # batch is a Batch object from PyG
        # Convert to dict for GraphModel
        batch = batch.to(device)
        
        # Convert Batch to dict
        batch_dict = {key: batch[key] for key in batch.keys}
        
        # The model computes forces via autograd, so positions need requires_grad
        if AtomicDataDict.POSITIONS_KEY in batch_dict:
            batch_dict[AtomicDataDict.POSITIONS_KEY] = batch_dict[AtomicDataDict.POSITIONS_KEY].requires_grad_(True)
        
        # Forward pass
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
            
            # Per atom energy
            n_atoms = torch.bincount(batch_dict[AtomicDataDict.BATCH_KEY]).float()
            e_per_atom_error = torch.abs((energy_pred - energy_true) / n_atoms.unsqueeze(-1))
            total_e_per_atom_error += e_per_atom_error.sum().item()
        
        # Clean up gradients
        if AtomicDataDict.POSITIONS_KEY in batch_dict and batch_dict[AtomicDataDict.POSITIONS_KEY].grad is not None:
            batch_dict[AtomicDataDict.POSITIONS_KEY].grad.zero_()
    
    # Compute averages
    metrics = {
        'f_mae': total_f_error / total_force_count if total_force_count > 0 else 0.0,
        'f_rmse': np.sqrt(total_f_sq_error / total_force_count) if total_force_count > 0 else 0.0,
        'e_mae': total_e_error / total_energy_count if total_energy_count > 0 else 0.0,
        'e_per_atom_mae': total_e_per_atom_error / total_energy_count if total_energy_count > 0 else 0.0,
    }
    
    # Simple loss estimate (force MAE + energy MAE)
    metrics['loss'] = metrics['f_mae'] + metrics['e_mae']
    
    return metrics


def count_parameters(model):
    """Count total number of parameters in model."""
    return sum(p.numel() for p in model.parameters())


def apply_rank_reduction(model, target_ranks, config):
    """
    Apply rl_orthogonal to reduce the rank of TT-cores in the model.
    
    Args:
        model: the model to modify (modified in-place)
        target_ranks: list of target ranks [r1, r2, r3]
        config: config dict with model parameters
    
    Returns:
        actual_ranks: the actual ranks after reduction
    """
    # Get the cores from the model
    cores = model.get_submodule('model.model.func.etn.cores')
    
    # Get instructions for each dimension
    instructions = []
    for i in range(config['d']):
        instr_tensor = model.get_buffer(f'model.model.func.etn.instructions_list_{i}')
        instructions.append([tuple(el) for el in instr_tensor.tolist()])
    
    # Current ranks [1, r1, r2, r3, 1]
    current_ranks = [1] + model.get_buffer(f'model.model.func.etn.N_rank_ett').tolist() + [1]
    
    # Target ranks [1, r1, r2, r3, 1]
    new_ranks = [1] + target_ranks + [1]
    
    print(f"  Applying rank reduction: {current_ranks[1:-1]} -> {target_ranks}")
    
    # Apply rl_orthogonal
    cores_list = [cores[i] for i in range(config['d'])]
    cores_new, actual_ranks = rl_orthogonal(cores_list, new_ranks, instructions)
    
    # Update the cores in the model
    for i in range(config['d']):
        model.get_submodule('model.model.func.etn.cores')[i] = cores_new[i]
    
    # Update the rank buffer in the model
    actual_ranks_tensor = torch.tensor(actual_ranks[1:-1], dtype=torch.long)
    model.get_buffer('model.model.func.etn.N_rank_ett').copy_(actual_ranks_tensor)
    
    print(f"  Actual ranks after reduction: {actual_ranks[1:-1]}")
    
    return actual_ranks[1:-1]


def main():
    """Main evaluation script."""
    
    # Configuration
    config_path = './config/example_ETN_opt_MEA.yaml'
    model_dir = './results/MEA_Allegro_0/example'
    model_name = 'best_model.pth'
    output_file = 'rank_reduction_results.csv'
    
    # Ranks to test (including original)
    test_ranks = [
        [16, 16, 16],  # Original - just orthogonalization
        [12, 12, 12],
        [8, 8, 8],
        [6, 6, 6],
        [4, 4, 4],
        [2, 2, 2],
    ]
    
    print("="*80)
    print("ETN Model Rank Reduction Evaluation")
    print("="*80)
    
    # Load config
    print("\nLoading configuration...")
    config = Config.from_file(config_path, defaults=default_config)
    device = torch.device(config['device'])
    print(f"  Using device: {device}")
    
    # Load datasets
    print("\nLoading datasets...")
    config['root'] = './results/MEA_Allegro_0'
    full_dataset = dataset_from_config(config, prefix="dataset")
    
    # Split into train and validation based on config
    n_train = config['n_train']
    n_val = config['n_val']
    
    print(f"  Total samples: {len(full_dataset)}")
    print(f"  Train samples: {n_train}")
    print(f"  Val samples: {n_val}")
    
    # Create indices for train/val split
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
    
    # Create data loaders
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    
    batch_size = config.get('batch_size', 10)
    collater = Collater.for_dataset(full_dataset)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collater,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collater,
    )
    
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    
    # Results storage
    results = []
    
    # First, evaluate model BEFORE any orthogonalization (baseline)
    print("\n" + "="*80)
    print("BASELINE: Evaluating model BEFORE orthogonalization")
    print("="*80)
    
    print("\nLoading model...")
    baseline_model = model_from_config(
        config=config, 
        initialize=True,
        dataset=full_dataset
    )
    
    # Load the trained weights
    state_dict = torch.load(f"{model_dir}/{model_name}", map_location=device)
    baseline_model.load_state_dict(state_dict)
    baseline_model.to(device)
    baseline_model.eval()
    
    # Disable gradient computation for all parameters
    for param in baseline_model.parameters():
        param.requires_grad = False
    
    # Count initial parameters
    n_params_baseline = count_parameters(baseline_model)
    print(f"  Parameters: {n_params_baseline:,}")
    
    # Get original ranks from the model
    original_ranks = baseline_model.get_buffer(f'model.model.func.etn.N_rank_ett').tolist()
    print(f"  Original ranks: {original_ranks}")
    
    # Evaluate on train set
    print("\n  Evaluating on train set...")
    baseline_train_metrics = compute_metrics(baseline_model, train_loader, device)
    print(f"    Train F MAE: {baseline_train_metrics['f_mae']:.6f}")
    print(f"    Train F RMSE: {baseline_train_metrics['f_rmse']:.6f}")
    print(f"    Train E MAE: {baseline_train_metrics['e_mae']:.6f}")
    print(f"    Train E/N MAE: {baseline_train_metrics['e_per_atom_mae']:.6f}")
    
    # Evaluate on validation set
    print("\n  Evaluating on validation set...")
    baseline_val_metrics = compute_metrics(baseline_model, val_loader, device)
    print(f"    Val F MAE: {baseline_val_metrics['f_mae']:.6f}")
    print(f"    Val F RMSE: {baseline_val_metrics['f_rmse']:.6f}")
    print(f"    Val E MAE: {baseline_val_metrics['e_mae']:.6f}")
    print(f"    Val E/N MAE: {baseline_val_metrics['e_per_atom_mae']:.6f}")
    
    # Store baseline results
    results.append({
        'rank_config': f'Baseline {original_ranks}',
        'n_params': n_params_baseline,
        'compression_ratio': 1.0,
        'train_f_mae': baseline_train_metrics['f_mae'],
        'train_f_rmse': baseline_train_metrics['f_rmse'],
        'train_e_mae': baseline_train_metrics['e_mae'],
        'train_e_per_atom_mae': baseline_train_metrics['e_per_atom_mae'],
        'train_loss': baseline_train_metrics['loss'],
        'val_f_mae': baseline_val_metrics['f_mae'],
        'val_f_rmse': baseline_val_metrics['f_rmse'],
        'val_e_mae': baseline_val_metrics['e_mae'],
        'val_e_per_atom_mae': baseline_val_metrics['e_per_atom_mae'],
        'val_loss': baseline_val_metrics['loss'],
    })
    
    # Clean up baseline model
    del baseline_model
    torch.cuda.empty_cache() if device.type == 'cuda' else None
    
    # Test each rank with orthogonalization
    for rank_config in test_ranks:
        print("\n" + "="*80)
        print(f"Testing rank configuration WITH orthogonalization: {rank_config}")
        print("="*80)
        
        # Load fresh model for each test
        print("\nLoading model...")
        # Build model from scratch with the dataset first, then load weights
        model = model_from_config(
            config=config, 
            initialize=True,
            dataset=full_dataset
        )
        
        # Load the trained weights
        state_dict = torch.load(f"{model_dir}/{model_name}", map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        
        # Disable gradient computation for all parameters
        for param in model.parameters():
            param.requires_grad = False
        
        # Count initial parameters
        n_params_before = count_parameters(model)
        print(f"  Parameters before: {n_params_before:,}")
        
        # Apply rank reduction
        actual_ranks = apply_rank_reduction(model, rank_config, config)
        
        # Count parameters after
        n_params_after = count_parameters(model)
        print(f"  Parameters after: {n_params_after:,}")
        print(f"  Compression ratio: {n_params_before / n_params_after:.2f}x")
        
        # Evaluate on train set
        print("\n  Evaluating on train set...")
        train_metrics = compute_metrics(model, train_loader, device)
        print(f"    Train F MAE: {train_metrics['f_mae']:.6f}")
        print(f"    Train F RMSE: {train_metrics['f_rmse']:.6f}")
        print(f"    Train E MAE: {train_metrics['e_mae']:.6f}")
        print(f"    Train E/N MAE: {train_metrics['e_per_atom_mae']:.6f}")
        
        # Evaluate on validation set
        print("\n  Evaluating on validation set...")
        val_metrics = compute_metrics(model, val_loader, device)
        print(f"    Val F MAE: {val_metrics['f_mae']:.6f}")
        print(f"    Val F RMSE: {val_metrics['f_rmse']:.6f}")
        print(f"    Val E MAE: {val_metrics['e_mae']:.6f}")
        print(f"    Val E/N MAE: {val_metrics['e_per_atom_mae']:.6f}")
        
        # Store results
        results.append({
            'rank_config': f'After ortho {str(actual_ranks)}',
            'n_params': n_params_after,
            'compression_ratio': n_params_before / n_params_after,
            'train_f_mae': train_metrics['f_mae'],
            'train_f_rmse': train_metrics['f_rmse'],
            'train_e_mae': train_metrics['e_mae'],
            'train_e_per_atom_mae': train_metrics['e_per_atom_mae'],
            'train_loss': train_metrics['loss'],
            'val_f_mae': val_metrics['f_mae'],
            'val_f_rmse': val_metrics['f_rmse'],
            'val_e_mae': val_metrics['e_mae'],
            'val_e_per_atom_mae': val_metrics['e_per_atom_mae'],
            'val_loss': val_metrics['loss'],
        })
    
    # Save results to CSV
    print("\n" + "="*80)
    print(f"Saving results to {output_file}")
    print("="*80)
    
    fieldnames = [
        'rank_config', 'n_params', 'compression_ratio',
        'train_f_mae', 'train_f_rmse', 'train_e_mae', 'train_e_per_atom_mae', 'train_loss',
        'val_f_mae', 'val_f_rmse', 'val_e_mae', 'val_e_per_atom_mae', 'val_loss',
    ]
    
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    
    print(f"\nResults saved successfully!")
    print(f"\nSummary:")
    print("-" * 80)
    print(f"{'Rank':<15} {'Params':<12} {'Compress':<10} {'Train F MAE':<15} {'Val F MAE':<15}")
    print("-" * 80)
    for r in results:
        print(f"{r['rank_config']:<15} {r['n_params']:<12,} "
              f"{r['compression_ratio']:<10.2f} "
              f"{r['train_f_mae']:<15.6f} {r['val_f_mae']:<15.6f}")
    print("-" * 80)


if __name__ == "__main__":
    main()

