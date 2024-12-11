#!/bin/bash
#SBATCH -p pgpu                    # Partition name
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:4  # Request 4 A100 GPUs
#SBATCH -t 24:00:00                # Time limit (24 hours)
#SBATCH --mem=400GB                # Memory limit
#SBATCH --output=job_output_%j.log # Log file name (%j expands to job ID)
#SBATCH --error=job_error_%j.log   # Error file name (%j expands to job ID)
#SBATCH --job-name=llm2vec_job     # Job name

# Activate the Conda environment
source activate llm2vec-repro

# Navigate to the project directory
cd /home/sthe14/llm2vec-repro/

# Run the commands
torchrun --nproc_per_node=4 experiments/run_supervised.py train_configs/cont-medical/MetaLlama3.1-mimicdi.json
torchrun --nproc_per_node=4 experiments/run_supervised.py train_configs/cont-medical/MetaLlama3.1-mednli.json