#!/bin/bash
# Training script for Allegro model on NbMoTaW dataset

# Activate virtual environment if needed
# source allegro_new/bin/activate

# Set environment variables for optimal performance
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Create output directory with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="allegro_training_${TIMESTAMP}"

echo "=========================================="
echo "Allegro Training Script for NbMoTaW"
echo "=========================================="
echo "Output directory: ${OUTPUT_DIR}"
echo "Configuration: train_allegro_nbmotaw.yaml"
echo "Dataset: benchmark_data/NbMoTaW.xyz"
echo "=========================================="
echo ""

# Run training
nequip-train \
    --config-name train_allegro_nbmotaw \
    hydra.run.dir=${OUTPUT_DIR}

echo ""
echo "=========================================="
echo "Training completed!"
echo "Results saved in: ${OUTPUT_DIR}"
echo "=========================================="

