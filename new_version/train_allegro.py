#!/usr/bin/env python3
"""
Training script for Allegro model on NbMoTaW dataset
Python alternative to the bash script
"""

import os
import sys
from datetime import datetime
from pathlib import Path


def main():
    """Run Allegro training on NbMoTaW dataset"""
    
    # Configuration
    config_file = "train_allegro_nbmotaw.yaml"
    dataset_file = "benchmark_data/NbMoTaW.xyz"
    
    # Check if files exist
    if not Path(config_file).exists():
        print(f"Error: Configuration file '{config_file}' not found!")
        sys.exit(1)
    
    if not Path(dataset_file).exists():
        print(f"Error: Dataset file '{dataset_file}' not found!")
        sys.exit(1)
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"allegro_training_{timestamp}"
    
    print("=" * 60)
    print("Allegro Training Script for NbMoTaW")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Configuration: {config_file}")
    print(f"Dataset: {dataset_file}")
    print("=" * 60)
    print()
    
    # Set environment variables for optimal performance
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    
    # Import and run training
    try:
        # Option 1: Using subprocess to call nequip-train
        import subprocess
        
        cmd = [
            "nequip-train",
            "--config-name", "train_allegro_nbmotaw",
            f"hydra.run.dir={output_dir}",
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        print()
        
        result = subprocess.run(cmd, check=True)
        
        print()
        print("=" * 60)
        print("Training completed successfully!")
        print(f"Results saved in: {output_dir}")
        print("=" * 60)
        
        return result.returncode
        
    except subprocess.CalledProcessError as e:
        print()
        print("=" * 60)
        print(f"Training failed with error code {e.returncode}")
        print("=" * 60)
        sys.exit(e.returncode)
    
    except ImportError as e:
        print()
        print("=" * 60)
        print("Error: NequIP/Allegro packages not found!")
        print("Please install the required packages:")
        print("  pip install nequip")
        print("  pip install allegro")
        print("Or activate your virtual environment:")
        print("  source allegro_new/bin/activate")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()

