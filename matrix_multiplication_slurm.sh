#!/bin/bash
#SBATCH --job-name=MATRIX
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4        # spawn 4 tasks (one per GPU)
#SBATCH --gres=gpu:4               # request 4 GPUs on the node
#SBATCH --cpus-per-task=4          # threads per process (adjust if needed)
#SBATCH --mem=0                     # let Slurm auto-assign RAM
#SBATCH --time=24:00:00            # walltime, adjust as needed
#SBATCH --reservation=sup-13563
#SBATCH --partition=ghx4 # e.g. gpu, gpu-long, etc.
#SBATCH --output=./logs/matrix_multiplication.out
#SBATCH --error=./logs/matrix_multiplication.err
#SBATCH --account=beok-dtai-gh    # Account name (adjust to your account)
#SBATCH --mail-user=rqzhang@berkeley.edu  # Email address to receive notifications
#SBATCH --mail-type=BEGIN,END,FAIL         # Send email at begin, end, or fail of job

# load your environment
module load cuda/12.6.1
module load gcc/11.4.0

# run the Python script (mp.spawn inside will fork per GPU)
srun python matrix_multiplication.py
