"""
HMM data generation and forward algorithm optimal estimator.

This module implements:
1. Random HMM generation (transition T and emission E matrices from Dirichlet priors)
2. HMM sequence generation (hidden states + observations)
3. Forward algorithm: exact Bayesian next-observation predictor P(o_{t+1} | o_1,...,o_t)
4. Evaluation utilities for comparing model predictions to the forward algorithm optimal
"""

import numpy as np
import torch
import torch.nn.functional as F
from contextlib import nullcontext


# ============================================================================
# HMM Data Generation
# ============================================================================

def sample_hmm_params(M, L, beta, batch_size, generator, device, dtype):
    """
    Sample random HMM parameters from Dirichlet priors.

    Args:
        M: number of hidden states
        L: number of observation symbols
        beta: Dirichlet concentration parameter (beta=1 gives uniform)
        batch_size: number of independent HMMs to sample
        generator: torch random generator
        device: torch device
        dtype: torch dtype

    Returns:
        T: (batch_size, M, M) transition matrices, T[b, i, j] = P(s_{t+1}=j | s_t=i)
        E: (batch_size, M, L) emission matrices, E[b, i, l] = P(o_t=l | s_t=i)
        pi: (batch_size, M) initial state distribution (uniform)
    """
    # Sample transition matrix: each row from Dirichlet(beta * 1_M)
    # Use Gamma distribution to construct Dirichlet
    alpha_T = beta * torch.ones(batch_size, M, M, device=device, dtype=dtype)
    gamma_T = torch._standard_gamma(alpha_T)
    T = gamma_T / gamma_T.sum(dim=-1, keepdim=True)

    # Sample emission matrix: each row from Dirichlet(beta * 1_L)
    alpha_E = beta * torch.ones(batch_size, M, L, device=device, dtype=dtype)
    gamma_E = torch._standard_gamma(alpha_E)
    E = gamma_E / gamma_E.sum(dim=-1, keepdim=True)

    # Uniform initial distribution
    pi = torch.ones(batch_size, M, device=device, dtype=dtype) / M

    return T, E, pi


def generate_hmm_sequences(T, E, pi, seq_length, generator, device):
    """
    Generate observation sequences from HMMs.

    Args:
        T: (batch_size, M, M) transition matrices
        E: (batch_size, M, L) emission matrices
        pi: (batch_size, M) initial state distributions
        seq_length: length of observation sequence to generate
        generator: torch random generator
        device: torch device

    Returns:
        observations: (batch_size, seq_length) observation indices (0..L-1)
        hidden_states: (batch_size, seq_length) hidden state indices (0..M-1) [for debugging only]
    """
    batch_size = T.shape[0]
    M = T.shape[1]

    observations = torch.zeros(batch_size, seq_length, device=device, dtype=torch.long)
    hidden_states = torch.zeros(batch_size, seq_length, device=device, dtype=torch.long)

    # Sample initial hidden state from pi
    states = torch.multinomial(pi, 1, replacement=True, generator=generator).squeeze(-1)  # (batch_size,)
    hidden_states[:, 0] = states

    # Sample initial observation from E[state]
    batch_idx = torch.arange(batch_size, device=device)
    emission_probs = E[batch_idx, states]  # (batch_size, L)
    observations[:, 0] = torch.multinomial(emission_probs, 1, replacement=True, generator=generator).squeeze(-1)

    # Generate sequence
    for t in range(1, seq_length):
        # Transition: sample s_t from T[s_{t-1}, :]
        trans_probs = T[batch_idx, states]  # (batch_size, M)
        states = torch.multinomial(trans_probs, 1, replacement=True, generator=generator).squeeze(-1)
        hidden_states[:, t] = states

        # Emission: sample o_t from E[s_t, :]
        emission_probs = E[batch_idx, states]  # (batch_size, L)
        observations[:, t] = torch.multinomial(emission_probs, 1, replacement=True, generator=generator).squeeze(-1)

    return observations, hidden_states


def get_hmm_batch(M, L, beta, seq_length, batch_size, generator, device, dtype):
    """
    Generate a batch of HMM observation sequences with fresh random HMMs.
    Each sequence in the batch comes from a DIFFERENT random HMM (ICL setting).

    Returns:
        x: (batch_size, seq_length) input observations (o_1, ..., o_T)
        y: (batch_size, seq_length) target observations (o_2, ..., o_{T+1})
        T_mat: (batch_size, M, M) transition matrices (for computing optimal)
        E_mat: (batch_size, M, L) emission matrices (for computing optimal)
        pi: (batch_size, M) initial distributions
    """
    # Sample fresh HMM for each sequence in batch
    T_mat, E_mat, pi = sample_hmm_params(M, L, beta, batch_size, generator, device, dtype)

    # Generate sequences of length seq_length + 1 (to get input/target pairs)
    obs, _ = generate_hmm_sequences(T_mat, E_mat, pi, seq_length + 1, generator, device)

    x = obs[:, :seq_length]   # input:  o_1, ..., o_T
    y = obs[:, 1:seq_length+1]  # target: o_2, ..., o_{T+1}

    return x, y, T_mat, E_mat, pi


# ============================================================================
# Forward Algorithm: Optimal Bayesian Predictor
# ============================================================================

def forward_algorithm_predict(T, E, pi, observations):
    """
    Compute the Bayes-optimal next-observation prediction using the forward algorithm.
    Given ground-truth T, E, and observation sequence o_1,...,o_t, computes:
        P(o_{t+1} | o_1,...,o_t) for all t

    This is the HMM analog of Laplacian smoothing in Bondaschi et al.

    Args:
        T: (batch_size, M, M) transition matrices
        E: (batch_size, M, L) emission matrices
        pi: (batch_size, M) initial state distributions
        observations: (batch_size, seq_length) observation sequence

    Returns:
        pred_probs: (batch_size, seq_length, L) predicted probability of each observation
                    pred_probs[:, t, :] = P(o_{t+1} = . | o_1, ..., o_{t+1})
                    Note: pred_probs[:, t, :] uses observations up to and including t
                    to predict the NEXT observation at position t+1.
                    This aligns with the model's prediction at position t.
    """
    batch_size, seq_length = observations.shape
    M = T.shape[1]
    L = E.shape[2]
    device = observations.device
    dtype = T.dtype

    pred_probs = torch.zeros(batch_size, seq_length, L, device=device, dtype=dtype)

    # Initialize belief state: alpha_0(j) = pi(j) * E(j, o_0)
    batch_idx = torch.arange(batch_size, device=device)
    obs_t = observations[:, 0]  # (batch_size,)

    # E_obs[b, j] = E[b, j, o_0[b]] — emission probability for each hidden state
    E_obs = E[batch_idx.unsqueeze(1).expand(-1, M),
              torch.arange(M, device=device).unsqueeze(0).expand(batch_size, -1),
              obs_t.unsqueeze(1).expand(-1, M)]  # (batch_size, M)
    alpha = pi * E_obs

    # Normalize
    alpha = alpha / (alpha.sum(dim=-1, keepdim=True) + 1e-30)

    # Predict next observation from current belief
    # P(o_{t+1} = k | o_1,...,o_t) = sum_j alpha_t(j) * sum_i T(j,i) * E(i, k)
    # = sum_j alpha_t(j) * [T @ E]_{j,k}
    # where [T @ E] is (M, L) matrix
    TE = torch.bmm(T, E)  # (batch_size, M, L): TE[b,j,k] = sum_i T[b,j,i] * E[b,i,k]
    pred = torch.bmm(alpha.unsqueeze(1), TE).squeeze(1)  # (batch_size, L)
    pred_probs[:, 0, :] = pred

    # Forward pass for t = 1, ..., seq_length - 1
    for t in range(1, seq_length):
        obs_t = observations[:, t]  # (batch_size,)

        # Predict step: alpha_pred(j) = sum_i alpha_{t-1}(i) * T(i, j)
        alpha_pred = torch.bmm(alpha.unsqueeze(1), T).squeeze(1)  # (batch_size, M)

        # Update step: alpha_t(j) = alpha_pred(j) * E(j, o_t)
        # E_obs[b, j] = E[b, j, o_t[b]]
        e_obs = E[batch_idx.unsqueeze(1).expand(-1, M),
                  torch.arange(M, device=device).unsqueeze(0).expand(batch_size, -1),
                  obs_t.unsqueeze(1).expand(-1, M)]

        alpha = alpha_pred * e_obs

        # Normalize
        alpha_sum = alpha.sum(dim=-1, keepdim=True) + 1e-30
        alpha = alpha / alpha_sum

        # Predict next observation
        pred = torch.bmm(alpha.unsqueeze(1), TE).squeeze(1)  # (batch_size, L)
        pred_probs[:, t, :] = pred

    return pred_probs


def optimal_hmm_loss(M, L, beta, seq_length, batch_size, generator, device, dtype, n_batches=16):
    """
    Compute the optimal loss achievable by the forward algorithm on random HMMs.
    Averages over n_batches of sequences.

    Returns:
        avg_loss: scalar, average cross-entropy loss of the forward algorithm predictor
    """
    total_loss = 0.0
    for _ in range(n_batches):
        x, y, T_mat, E_mat, pi = get_hmm_batch(M, L, beta, seq_length, batch_size, generator, device, dtype)

        # Get optimal predictions
        pred_probs = forward_algorithm_predict(T_mat, E_mat, pi, x)

        # Compute cross-entropy loss
        # pred_probs[:, t, :] predicts y[:, t]
        log_probs = torch.log(pred_probs + 1e-30)
        loss = F.nll_loss(log_probs.reshape(-1, log_probs.size(-1)), y.reshape(-1))
        total_loss += loss.item()

    return total_loss / n_batches


# ============================================================================
# Evaluation Utilities
# ============================================================================

@torch.no_grad()
def eval_hmm(model, M, L, beta, seq_length, batch_size, generator, device, dtype,
             max_num_batches=24, ctx=nullcontext()):
    """
    Evaluate model on random HMM sequences.
    Returns accuracy, loss, perplexity, and gap from forward algorithm optimal.
    """
    assert model.training == False

    loss_list, acc_list, opt_loss_list = [], [], []

    for _ in range(max_num_batches):
        x, y, T_mat, E_mat, pi = get_hmm_batch(M, L, beta, seq_length, batch_size, generator, device, dtype)

        with ctx:
            outputs = model(x, targets=y)

        loss_list.append(outputs['loss'].item())
        acc_list.append((outputs['logits'].argmax(-1) == y).float().mean().item())

        # Compute optimal loss for this batch
        pred_probs = forward_algorithm_predict(T_mat, E_mat, pi, x)
        log_probs = torch.log(pred_probs + 1e-30)
        opt_loss = F.nll_loss(log_probs.reshape(-1, log_probs.size(-1)), y.reshape(-1))
        opt_loss_list.append(opt_loss.item())

    val_loss = np.mean(loss_list)
    val_acc = np.mean(acc_list)
    val_perplexity = np.exp(val_loss)
    opt_loss = np.mean(opt_loss_list)

    return val_acc, val_loss, val_perplexity, opt_loss


@torch.no_grad()
def eval_hmm_probs(model, M, L, beta, seq_length, generator, device, dtype, ctx=nullcontext()):
    """
    On a FIXED test sequence, compare model's predicted probabilities to the
    forward algorithm's optimal predictions. This produces the analog of
    Bondaschi's Figure 1.

    Returns:
        model_probs: (seq_length, L) model predicted probabilities
        optimal_probs: (seq_length, L) forward algorithm predicted probabilities
        observations: (seq_length,) the test observation sequence
    """
    assert model.training == False

    # Generate a single test sequence
    x, y, T_mat, E_mat, pi = get_hmm_batch(M, L, beta, seq_length, 1, generator, device, dtype)

    # Model predictions
    with ctx:
        outputs = model(x, targets=y)
    model_probs = F.softmax(outputs['logits'], dim=-1).squeeze(0)  # (seq_length, L)

    # Forward algorithm predictions
    opt_probs = forward_algorithm_predict(T_mat, E_mat, pi, x).squeeze(0)  # (seq_length, L)

    return model_probs, opt_probs, x.squeeze(0), y.squeeze(0), T_mat.squeeze(0), E_mat.squeeze(0)


@torch.no_grad()
def extract_at_values(model, M, L, beta, seq_length, generator, device, dtype, ctx=nullcontext()):
    """
    Extract the state-to-state transition factor a_t = exp(A * dt) across positions.
    This is the critical diagnostic: Bondaschi found a_t ≈ 1 for Markov chains.
    For HMMs, we want to see if a_t shows novel structure.

    Returns:
        at_values: (seq_length,) the value of a_t at each position
        dt_values: (seq_length,) the raw dt values
        A_value: scalar, the learned A parameter
    """
    assert model.training == False

    # Generate a test sequence
    x, y, T_mat, E_mat, pi = get_hmm_batch(M, L, beta, seq_length, 1, generator, device, dtype)

    # Get the Mamba layer
    # Navigate: model -> layers[0] -> mixer (Mamba2)
    raw_model = model
    if hasattr(model, '_orig_mod'):
        raw_model = model._orig_mod

    mamba_layer = raw_model.layers[0].mixer

    # Forward pass through embedding
    hidden = raw_model.embedding(x)

    # Get in_proj output
    if hasattr(raw_model.layers[0].config, 'layernorm') and raw_model.layers[0].config.layernorm:
        hidden = raw_model.layers[0].norm(hidden).to(dtype)

    zxbcdt = mamba_layer.in_proj(hidden)
    A = -torch.exp(mamba_layer.A_log).to(dtype)

    z, xBC, dt = torch.split(
        zxbcdt,
        [mamba_layer.d_inner,
         mamba_layer.d_inner + 2 * mamba_layer.ngroups * mamba_layer.d_state,
         mamba_layer.nheads],
        dim=-1
    )

    dt = F.softplus(dt + mamba_layer.dt_bias).to(dtype)  # (1, seq_length, nheads)

    # a_t = exp(A * dt) where A is negative, so a_t in (0, 1)
    At = torch.exp(A * dt)  # (1, seq_length, nheads)

    at_values = At.squeeze(0).squeeze(-1).cpu().numpy()  # (seq_length,) if nheads=1
    dt_values = dt.squeeze(0).squeeze(-1).cpu().numpy()
    A_value = A.item() if A.numel() == 1 else A.cpu().numpy()

    return at_values, dt_values, A_value, x.squeeze(0).cpu().numpy()


def save_checkpoint(model, opt, scheduler, itr, ckpt_path, **extra_args):
    checkpoint = dict({
        'model': model.state_dict(),
        'optimizer': opt.state_dict(),
        'scheduler': scheduler.state_dict(),
        'itr': itr,
    }, **extra_args)
    torch.save(checkpoint, ckpt_path)
