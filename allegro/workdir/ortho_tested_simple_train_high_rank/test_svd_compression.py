"""
Compression analysis using SVD-based rank reduction.

This script:
1. Loads the trained ETN model
2. Applies SVD rank reduction to various target ranks
3. Evaluates accuracy on train/val sets WITHOUT retraining
4. Computes compression ratios
5. Generates analysis plots and tables
"""

import sys
import os
import csv
from pathlib import Path
import torch
import numpy as np

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nequip.data import dataset_from_config, DataLoader, Collater
from nequip.utils import Config
from nequip.data import AtomicDataDict
from nequip.model import model_from_config
from rank_reduction_svd import rl_orthogonal_svd

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
        dict with keys: f_mae, f_rmse, e_mae, e_per_atom_mae, loss
    """
    model.eval()
    
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
    
    # Compute final metrics
    metrics = {
        'f_mae': total_f_error / total_force_count if total_force_count > 0 else 0,
        'f_rmse': np.sqrt(total_f_sq_error / total_force_count) if total_force_count > 0 else 0,
        'e_mae': total_e_error / total_energy_count if total_energy_count > 0 else 0,
        'e_per_atom_mae': total_e_per_atom_error / total_energy_count if total_energy_count > 0 else 0,
        'loss': 0,  # Not computing loss for simplicity
    }
    
    return metrics


def count_parameters(model):
    """Count trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters())


def main():
    # Configuration
    config_path = "results/MEA_Allegro_0/example/config.yaml"
    model_dir = "results/MEA_Allegro_0/example"
    model_name = "best_model.pth"
    
    # Thresholds (ratio σ / σ_max) used for zeroing singular values
    threshold_tests = [
        None,   # reference (no zeroing)
        5e-1,
        3e-1,
        2e-1,
        1e-1,
        5e-2,
        2e-2,
        1e-2,
        5e-3,
        1e-3,
    ]
    
    print("="*80)
    print("SVD-Based Rank Reduction Compression Analysis")
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
    
    # First, evaluate model BEFORE any rank reduction (baseline)
    print("\n" + "="*80)
    print("BASELINE: Original model (no compression)")
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
        'actual_ranks': str(original_ranks),
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
        'zeroed_svs': 'baseline',
        'threshold': 'baseline',
        'kept_svs': str(original_ranks),
        'zeroed_total': 0,
    })
    
    # Clean up baseline model
    del baseline_model
    torch.cuda.empty_cache() if device.type == 'cuda' else None
    
    # Test each threshold with SVD zeroing
    for threshold in threshold_tests:
        print("\n" + "="*80)
        if threshold is None:
            print("Testing SVD zero-out with threshold: None (no zeroing)")
        else:
            print(f"Testing SVD zero-out with threshold ratio: {threshold:.3e}")
        print("="*80)

        print("\nLoading model and applying zero-out compression...")
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
        for param in model.parameters():
            param.requires_grad = False

        # Count parameters (will remain unchanged with zero-out compression)
        n_params_current = count_parameters(model)
        print(f"  Model parameters: {n_params_current:,}")
        if threshold is None:
            print("  Zeroing smallest singular values per bond: [0, 0, 0]")
        else:
            print(f"  Zeroing threshold (σ/σ_max): {threshold}")

        # Apply SVD zero-out to get compressed cores
        cores = model.get_submodule('model.model.func.etn.cores')
        instructions = []
        for i in range(config['d']):
            instructions.append(
                [tuple(el) for el in model.get_buffer(f'model.model.func.etn.instructions_list_{i}').tolist()]
            )

        ranks = [1] + original_ranks + [1]

        cores_compressed, ranks_new, info = rl_orthogonal_svd(
            cores,
            ranks.copy(),
            instructions,
            target_ranks=None,
            threshold=threshold,
            zero_truncate=True,
            return_info=True,
        )

        kept_counts = info["kept_counts"]
        effective_ranks = kept_counts
        zeroed_counts = [
            r_orig - r_kept for r_orig, r_kept in zip(original_ranks, effective_ranks)
        ]
        zeroed_total = sum(zeroed_counts)
        actual_ranks = ranks_new[1:-1]
        print(f"  Effective ranks retained: {effective_ranks}")
        print(f"  Zeroed counts per bond: {zeroed_counts}")
        print(f"  Total singular values zeroed: {zeroed_total}")
        print(f"  Core tensor shapes preserved with ranks buffer: {actual_ranks}")

        # Load the compressed cores back into the model
        for i in range(config['d']):
            model.get_submodule('model.model.func.etn.cores')[i] = cores_compressed[i]

        # Parameters remain unchanged; compute effective compression ratio based on desired ranks
        n_params_after = count_parameters(model)
        compression = sum(effective_ranks) / sum(original_ranks)
        print(f"  Parameters after zero-out: {n_params_after:,} (physical params unchanged)")
        print(f"  Effective rank-based compression ratio: {compression:.3f}x")

        # Evaluate on train set
        print("\n  Evaluating on train set...")
        train_metrics = compute_metrics(model, train_loader, device)
        print(f"    Train F MAE: {train_metrics['f_mae']:.6f} (Δ: {train_metrics['f_mae'] - baseline_train_metrics['f_mae']:+.6f})")
        print(f"    Train F RMSE: {train_metrics['f_rmse']:.6f}")
        print(f"    Train E MAE: {train_metrics['e_mae']:.6f}")
        print(f"    Train E/N MAE: {train_metrics['e_per_atom_mae']:.6f} (Δ: {train_metrics['e_per_atom_mae'] - baseline_train_metrics['e_per_atom_mae']:+.6f})")
        
        # Evaluate on validation set
        print("\n  Evaluating on validation set...")
        val_metrics = compute_metrics(model, val_loader, device)
        print(f"    Val F MAE: {val_metrics['f_mae']:.6f} (Δ: {val_metrics['f_mae'] - baseline_val_metrics['f_mae']:+.6f})")
        print(f"    Val F RMSE: {val_metrics['f_rmse']:.6f}")
        print(f"    Val E MAE: {val_metrics['e_mae']:.6f}")
        print(f"    Val E/N MAE: {val_metrics['e_per_atom_mae']:.6f} (Δ: {val_metrics['e_per_atom_mae'] - baseline_val_metrics['e_per_atom_mae']:+.6f})")
        
        # Store results
        results.append({
            'rank_config': f'SVD_thresh {threshold if threshold is not None else "none"}',
            'actual_ranks': str(effective_ranks),
            'n_params': n_params_after,
            'compression_ratio': compression,
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
            'zeroed_svs': str(zeroed_counts),
            'threshold': threshold if threshold is not None else 'none',
            'kept_svs': str(effective_ranks),
            'zeroed_total': zeroed_total,
        })
        
        # Clean up model
        del model
        torch.cuda.empty_cache() if device.type == 'cuda' else None
    
    # Save results to CSV
    print("\n" + "="*80)
    print("Saving results...")
    print("="*80)
    
    output_file = "svd_compression_results.csv"
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    
    print(f"  Results saved to: {output_file}")
    
    # Print summary table
    print("\n" + "="*80)
    print("COMPRESSION ANALYSIS SUMMARY")
    print("="*80)
    print()
    print(f"{'Rank Config':<20} {'Params':<12} {'Comp.':<8} {'Train F MAE':<14} {'Val F MAE':<14} "
          f"{'Train E/N MAE':<14} {'Val E/N MAE':<14} {'Threshold':<12} "
          f"{'Kept':<18} {'Zeroed':<18} {'Zeroed Tot':<11}")
    print("-" * 195)
    
    for r in results:
        print(f"{r['rank_config']:<20} {r['n_params']:<12,} {r['compression_ratio']:<8.2f} "
              f"{r['train_f_mae']:<14.6f} {r['val_f_mae']:<14.6f} "
              f"{r['train_e_per_atom_mae']:<14.6f} {r['val_e_per_atom_mae']:<14.6f} "
              f"{r['threshold']:<12} {r['kept_svs']:<18} {r['zeroed_svs']:<18} "
              f"{r.get('zeroed_total', 0):<11}")
    
    print("\n" + "="*80)
    print("Analysis complete! Check svd_compression_results.csv for full results.")
    print("Run svd_compression_analysis.ipynb to visualize the results.")
    print("="*80)


if __name__ == "__main__":
    main()

