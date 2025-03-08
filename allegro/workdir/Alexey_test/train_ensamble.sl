#!/bin/bash


module load python
conda activate tdep
source /pscratch/sd/v/vladygin/doped-Si_project/MLFF_TDEP/testbench/bin/activate

python run_all.py 10 
