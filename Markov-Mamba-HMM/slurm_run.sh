#!/bin/bash
#SBATCH --job-name=hmm-mamba
#SBATCH --output=hmm_mamba_%A_%a.out
#SBATCH --error=hmm_mamba_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-2
#
# Submit with: sbatch slurm_run.sh
# This runs all 3 experiments as a job array.

# --- Adjust these paths for your HPC ---
# module load cuda/12.1   # uncomment if needed
# module load python/3.10 # uncomment if needed
# source activate myenv   # uncomment if using conda

cd Markov-Mamba-HMM/src

# Common args for MambaZero
COMMON="--config_format hmm \
  --hmm_M 2 --hmm_L 2 --hmm_beta 1.0 \
  --vocab_size 2 \
  --d_model 8 --d_state 8 --expand 2 \
  --nheads 1 --ngroups 1 \
  --sequence_length 128 \
  --batch_size 64 \
  --iterations 10000 --eval_freq 500 \
  --lr 2e-3 --scheduler cos --opt adamw \
  --conv --no_mlp \
  --results_base_folder ../exps \
  --seed 42"

case $SLURM_ARRAY_TASK_ID in
  0)
    echo "Experiment A: 1-layer MambaZero, w=2"
    python main.py $COMMON --n_layer 1 --d_conv 2 --wandb_run_prefix "expA_1layer_w2"
    ;;
  1)
    echo "Experiment B: 1-layer MambaZero, w=3"
    python main.py $COMMON --n_layer 1 --d_conv 3 --wandb_run_prefix "expB_1layer_w3"
    ;;
  2)
    echo "Experiment C: 2-layer MambaZero, w=2"
    python main.py $COMMON --n_layer 2 --d_conv 2 --wandb_run_prefix "expC_2layer_w2"
    ;;
esac

echo "Done."
