#!/bin/bash
#SBATCH -J NaBr_born
#SBATCH -A m2129_g
#SBATCH -q regular
#SBATCH -t 12:00:00        
#SBATCH -N 1      
#SBATCH -C gpu&hbm80g
#SBATCH --ntasks-per-node=1
#SBATCH -c 128
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none
#SBATCH --exclusive

#module load python/3.9

#conda activate /pscratch/sd/v/vladygin/NaBr_project/ALLEGRO_setup/new_setup/allegro_env/

#module load PrgEnv-gnu cray-mpich cudatoolkit craype-accel-nvidia80

module load conda
module load cudatoolkit
module load cudnn
module load craype-accel-nvidia80

MPICH_GPU_SUPPORT_ENABLED=1

srun -n 4 -G 4 --gpu-bind=none /pscratch/sd/v/vladygin/nequip-allegro-lammps/lammps/build/lmp -sf kk -k on g 4 -pk kokkos newton on neigh full -in si_rdf.in

#/pscratch/sd/v/vladygin/nequip-allegro-lammps/lammps/build/lmp -in si_rdf.in
