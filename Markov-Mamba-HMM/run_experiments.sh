#!/bin/bash
#
# HMM Mamba Experiments
# =====================
# Three pilot experiments to test whether Mamba learns the forward algorithm
# for Hidden Markov Models.
#
# Run from: Markov-Mamba-HMM/src/
# Usage: bash ../run_experiments.sh
#
# Each experiment: 10000 iterations, M=2 hidden states, L=2 observations
# MambaZero configuration: conv=yes, no_mlp=yes, no gate, no conv_act
# (This matches Bondaschi's MambaZero: only convolution + SSM, no gating/ReLU/MLP)

set -e

echo "============================================"
echo "HMM Mamba Pilot Experiments"
echo "============================================"

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

# ============================================
# Experiment A: 1-layer MambaZero, w=2
# ============================================
echo ""
echo ">>> Experiment A: 1-layer MambaZero, conv window w=2"
echo ""
python main.py $COMMON \
  --n_layer 1 \
  --d_conv 2 \
  --wandb_run_prefix "expA_1layer_w2"

# ============================================
# Experiment B: 1-layer MambaZero, w=3
# ============================================
echo ""
echo ">>> Experiment B: 1-layer MambaZero, conv window w=3"
echo ""
python main.py $COMMON \
  --n_layer 1 \
  --d_conv 3 \
  --wandb_run_prefix "expB_1layer_w3"

# ============================================
# Experiment C: 2-layer MambaZero, w=2
# ============================================
echo ""
echo ">>> Experiment C: 2-layer MambaZero, conv window w=2"
echo ""
python main.py $COMMON \
  --n_layer 2 \
  --d_conv 2 \
  --wandb_run_prefix "expC_2layer_w2"

echo ""
echo "============================================"
echo "All experiments complete!"
echo "Results saved to ../exps/hmm/base/"
echo "============================================"
echo ""
echo "Key output files per experiment:"
echo "  metrics.json         - loss gaps, a_t stats, L1 distances"
echo "  fig1_prob_vs_optimal.png - predicted probs vs forward algorithm"
echo "  fig4_at_values.png   - a_t across positions"
echo "  prob_curves.npz      - raw probability data"
echo "  at_values.npz        - raw a_t data"
