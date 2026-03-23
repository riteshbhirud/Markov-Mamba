"""
Training loop for HMM experiments.
Tracks:
  1. Training loss gap from forward algorithm optimal every eval_freq steps
  2. Predicted probability curve vs forward algorithm on fixed test sequence
  3. Value of a_t across positions at convergence
"""

from contextlib import nullcontext
import os
import json
import numpy as np
import torch
import torch.nn.functional as F
import time

from .utils import (
    get_hmm_batch, eval_hmm, eval_hmm_probs, extract_at_values,
    optimal_hmm_loss, save_checkpoint
)


def train_hmm(model, opt, scheduler, iterations, acc_steps, batch_size,
              sequence_length, M, L, beta, generator, eval_freq,
              ckpt_path, extra_args):
    """
    Main training loop for HMM experiments.
    """
    device = extra_args.device
    dtype = extra_args.dtype
    device_type = 'cuda' if 'cuda' in str(device) else 'cpu'
    type_ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(
        device_type=device_type, dtype=torch.float16)

    itr, substep = 0, 0
    results_dir = os.path.dirname(ckpt_path)

    if device_type == 'cuda':
        print(f"Compiling model ...")
        model = torch.compile(model)
    else:
        print("Skipping torch.compile on CPU")

    # Compute baseline optimal loss (forward algorithm average)
    print("Computing forward algorithm optimal loss (baseline)...")
    opt_loss_baseline = optimal_hmm_loss(
        M, L, beta, sequence_length, batch_size, generator, device, dtype, n_batches=16
    )
    print(f"Forward algorithm optimal loss: {opt_loss_baseline:.4f}")

    # Storage for tracking metrics
    metrics = {
        'iterations': [],
        'train_loss': [],
        'val_loss': [],
        'val_acc': [],
        'opt_loss': [],
        'loss_gap': [],
        'opt_loss_baseline': opt_loss_baseline,
    }

    # Use wandb if requested
    use_wandb = hasattr(extra_args, 'wandb') and extra_args.wandb
    if use_wandb:
        import wandb
        wandb.log({"val/opt_loss_baseline": opt_loss_baseline})

    model.train()
    t0 = time.time()

    while itr < iterations:
        for microstep_idx in range(acc_steps):
            x, y, T_mat, E_mat, pi = get_hmm_batch(
                M, L, beta, sequence_length, batch_size, generator, device, dtype
            )
            with type_ctx:
                outputs = model(x, targets=y)
            loss = outputs['loss'] / acc_steps
            loss.backward()
            substep += 1

        if extra_args.grad_clip != 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), extra_args.grad_clip)

        opt.step()
        scheduler.step()
        opt.zero_grad(set_to_none=True)
        itr += 1

        # ============ Evaluation ============
        if itr % eval_freq == 0 or itr == iterations:
            t1 = time.time()
            dt = t1 - t0

            model.eval()
            train_loss = loss.detach().cpu().item() * acc_steps  # undo the division

            val_acc, val_loss, val_perplexity, val_opt_loss = eval_hmm(
                model, M, L, beta, sequence_length, batch_size,
                generator, device, dtype, max_num_batches=10, ctx=type_ctx
            )

            loss_gap = val_loss - val_opt_loss

            # Record metrics
            metrics['iterations'].append(itr)
            metrics['train_loss'].append(train_loss)
            metrics['val_loss'].append(val_loss)
            metrics['val_acc'].append(val_acc)
            metrics['opt_loss'].append(val_opt_loss)
            metrics['loss_gap'].append(loss_gap)

            current_lr = scheduler.get_last_lr()[0] if scheduler is not None else extra_args.lr

            print_string = (
                f"{itr} [train] loss={train_loss:.4f} "
                f"[val] loss={val_loss:.4f}, opt={val_opt_loss:.4f}, "
                f"gap={loss_gap:.4f}, acc={val_acc:.4f}, pp={val_perplexity:.2f}"
            )
            print_string += f" [time per itr] {dt*1000/eval_freq:.2f}ms"
            print_string += f" [lr] {current_lr:.5f}"
            print(print_string)

            if use_wandb:
                wandb.log({
                    "iter": itr,
                    "train/loss": train_loss,
                    "val/loss": val_loss,
                    "val/opt_loss": val_opt_loss,
                    "val/loss_gap": loss_gap,
                    "val/acc": val_acc,
                    "val/perplexity": val_perplexity,
                    "lr": current_lr,
                })

            # ============ At convergence (final iteration): detailed analysis ============
            if itr == iterations:
                print("\n" + "="*60)
                print("FINAL ANALYSIS AT CONVERGENCE")
                print("="*60)

                # --- 1. Plot predicted probability vs forward algorithm optimal ---
                print("\n--- Predicted vs Optimal Probabilities ---")
                model_probs, opt_probs, obs_seq, target_seq, T_test, E_test = eval_hmm_probs(
                    model, M, L, beta, sequence_length, generator, device, dtype, ctx=type_ctx
                )

                # Compute L1 distance from optimal
                l1_dist = (model_probs - opt_probs).abs().sum(dim=-1).mean().item()
                print(f"Average L1 distance from forward algorithm optimal: {l1_dist:.6f}")

                # Save probability curves
                np.savez(
                    os.path.join(results_dir, 'prob_curves.npz'),
                    model_probs=model_probs.cpu().numpy(),
                    opt_probs=opt_probs.cpu().numpy(),
                    observations=obs_seq.cpu().numpy(),
                    targets=target_seq.cpu().numpy(),
                )

                # Plot Figure 1 analog
                _plot_prob_curves(
                    model_probs.cpu().numpy(),
                    opt_probs.cpu().numpy(),
                    obs_seq.cpu().numpy(),
                    L,
                    os.path.join(results_dir, 'fig1_prob_vs_optimal.png')
                )

                # --- 2. Extract and plot a_t values (Figure 4 analog) ---
                print("\n--- a_t Values Across Positions ---")
                at_vals, dt_vals, A_val, at_obs = extract_at_values(
                    model, M, L, beta, sequence_length, generator, device, dtype, ctx=type_ctx
                )

                print(f"Learned A value: {A_val}")
                print(f"a_t statistics: mean={at_vals.mean():.6f}, std={at_vals.std():.6f}, "
                      f"min={at_vals.min():.6f}, max={at_vals.max():.6f}")

                # Single-sequence breakdown by observation
                if L == 2:
                    at_obs0 = at_vals[at_obs == 0]
                    at_obs1 = at_vals[at_obs == 1]
                    if len(at_obs0) > 0 and len(at_obs1) > 0:
                        print(f"  [single seq] a_t when o_t=0: mean={at_obs0.mean():.6f}, std={at_obs0.std():.6f}")
                        print(f"  [single seq] a_t when o_t=1: mean={at_obs1.mean():.6f}, std={at_obs1.std():.6f}")
                        print(f"  [single seq] Difference in means: {abs(at_obs0.mean() - at_obs1.mean()):.6f}")

                # Aggregate a_t breakdown over MANY test sequences (the critical number)
                print("\n--- a_t by Observation Value (aggregated over 50 test sequences) ---")
                all_at_by_obs = {k: [] for k in range(L)}
                all_at_all = []
                for _ in range(50):
                    at_v, _, _, obs_v = extract_at_values(
                        model, M, L, beta, sequence_length, generator, device, dtype, ctx=type_ctx
                    )
                    all_at_all.append(at_v)
                    for k in range(L):
                        all_at_by_obs[k].append(at_v[obs_v == k])

                all_at_flat = np.concatenate(all_at_all)
                at_by_obs_means = {}
                at_by_obs_stds = {}
                for k in range(L):
                    vals = np.concatenate(all_at_by_obs[k])
                    at_by_obs_means[k] = float(vals.mean())
                    at_by_obs_stds[k] = float(vals.std())
                    print(f"  a_t when o_t={k}: mean={vals.mean():.8f}, std={vals.std():.8f}, n={len(vals)}")

                if L == 2:
                    obs_diff = abs(at_by_obs_means[0] - at_by_obs_means[1])
                    print(f"  >>> CRITICAL NUMBER: |mean_a_t(o=0) - mean_a_t(o=1)| = {obs_diff:.8f}")
                    # Also report relative to overall std
                    overall_std = float(all_at_flat.std())
                    if overall_std > 0:
                        print(f"  >>> Relative to overall a_t std ({overall_std:.8f}): "
                              f"ratio = {obs_diff / overall_std:.4f}")

                np.savez(
                    os.path.join(results_dir, 'at_values.npz'),
                    at_values=at_vals,
                    dt_values=dt_vals,
                    A_value=A_val,
                    observations=at_obs,
                    # Aggregated stats
                    at_all_flat=all_at_flat,
                    at_by_obs_means=np.array([at_by_obs_means[k] for k in range(L)]),
                    at_by_obs_stds=np.array([at_by_obs_stds[k] for k in range(L)]),
                )

                _plot_at_values(
                    at_vals, at_obs, L,
                    os.path.join(results_dir, 'fig4_at_values.png')
                )

                # --- 3. Compute multiple L1 distances across sequences ---
                print("\n--- L1 Distance Across Multiple Test Sequences ---")
                l1_distances = []
                for _ in range(10):
                    mp, op, _, _, _, _ = eval_hmm_probs(
                        model, M, L, beta, sequence_length, generator, device, dtype, ctx=type_ctx
                    )
                    l1_d = (mp - op).abs().sum(dim=-1).mean().item()
                    l1_distances.append(l1_d)
                l1_mean = np.mean(l1_distances)
                l1_std = np.std(l1_distances)
                print(f"L1 distance: {l1_mean:.6f} +/- {l1_std:.6f}")

                # Save all metrics
                metrics['final_l1_distance_mean'] = l1_mean
                metrics['final_l1_distance_std'] = l1_std
                metrics['at_mean'] = float(all_at_flat.mean())
                metrics['at_std'] = float(all_at_flat.std())
                metrics['A_value'] = float(A_val) if isinstance(A_val, (int, float)) else A_val.tolist()
                metrics['at_by_obs_means'] = at_by_obs_means
                metrics['at_by_obs_stds'] = at_by_obs_stds
                if L == 2:
                    metrics['at_obs_diff'] = abs(at_by_obs_means[0] - at_by_obs_means[1])

                # --- Summary ---
                print("\n" + "="*60)
                print("SUMMARY")
                print("="*60)
                print(f"1. Does Mamba converge to forward algorithm optimal?")
                print(f"   Final loss gap: {loss_gap:.6f}")
                print(f"   Final L1 distance: {l1_mean:.6f} +/- {l1_std:.6f}")
                converged = l1_mean < 0.1
                print(f"   Converged: {'YES' if converged else 'NO'}")
                print(f"\n2. What is a_t across positions?")
                print(f"   Mean (50 seqs): {all_at_flat.mean():.8f}")
                print(f"   Std  (50 seqs): {all_at_flat.std():.8f}")
                approx_one = all_at_flat.mean() > 0.95
                print(f"   Approximately 1 everywhere: {'YES' if approx_one else 'NO'}")
                print(f"\n3. Novel structure in a_t? (THE KEY QUESTION)")
                for k in range(L):
                    print(f"   mean a_t when o_t={k}: {at_by_obs_means[k]:.8f} (std={at_by_obs_stds[k]:.8f})")
                if L == 2:
                    obs_diff = abs(at_by_obs_means[0] - at_by_obs_means[1])
                    print(f"   |mean_a_t(o=0) - mean_a_t(o=1)| = {obs_diff:.8f}")
                    novel = obs_diff > 0.01
                    print(f"   Shows observation-dependent modulation: {'YES' if novel else 'NO'}")
                    if novel:
                        print(f"   >>> STRONG PAPER: a_t modulates with emissions")
                    else:
                        print(f"   >>> ACCEPTABLE PAPER: a_t behaves like Markov case")
                else:
                    overall_std = float(all_at_flat.std())
                    novel = overall_std > 0.01
                    print(f"   a_t overall std: {overall_std:.8f}")
                    print(f"   Shows positional variation: {'YES' if novel else 'NO'}")

            model.train()
            t0 = time.time()

    # Save final checkpoint and metrics
    print(f"\nSaving checkpoint to {ckpt_path}")
    save_checkpoint(model=model, opt=opt, scheduler=scheduler, itr=itr, ckpt_path=ckpt_path)

    with open(os.path.join(results_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2, default=str)

    print("Training complete.")


def _plot_prob_curves(model_probs, opt_probs, observations, L, save_path):
    """
    Plot analog of Bondaschi Figure 1:
    Model predicted probability vs forward algorithm optimal on a fixed test sequence.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        seq_len = model_probs.shape[0]
        fig, axes = plt.subplots(1, L, figsize=(7*L, 5))
        if L == 1:
            axes = [axes]

        for obs_val in range(L):
            ax = axes[obs_val]
            # For positions where current observation is obs_val, plot P(next_obs=1)
            # (or plot P(next_obs=obs_val) more generally)
            mask = observations == obs_val
            positions = np.where(mask)[0]

            if len(positions) > 0:
                # Plot model's P(next=1 | current=obs_val)
                if L == 2:
                    ax.plot(range(len(positions)), model_probs[positions, 1],
                            'o', markersize=2, alpha=0.6, label='Mamba', color='orange')
                    ax.plot(range(len(positions)), opt_probs[positions, 1],
                            'o', markersize=2, alpha=0.6, label='Forward alg. optimal', color='green')
                else:
                    for k in range(L):
                        ax.plot(range(len(positions)), model_probs[positions, k],
                                'o', markersize=2, alpha=0.6, label=f'Mamba P(o={k})')
                        ax.plot(range(len(positions)), opt_probs[positions, k],
                                '--', alpha=0.6, label=f'Optimal P(o={k})')

            ax.set_xlabel(f't : o_t = {obs_val}')
            ax.set_ylabel('Predicted probability')
            ax.set_title(f'Next-token prediction (o_t = {obs_val})')
            ax.legend()
            ax.set_ylim(-0.05, 1.05)

        plt.suptitle('Mamba vs Forward Algorithm Optimal (HMM)')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved probability curve plot to {save_path}")
    except Exception as e:
        print(f"Could not save plot: {e}")


def _plot_at_values(at_values, observations, L, save_path):
    """
    Plot analog of Bondaschi Figure 4: a_t across positions.
    Color-code by observation value to see if a_t varies with emissions.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: a_t across all positions
        ax = axes[0]
        ax.plot(at_values, linewidth=0.8, color='blue', alpha=0.7)
        ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='a_t = 1')
        ax.set_xlabel('Position t')
        ax.set_ylabel('a_t = exp(A * dt)')
        ax.set_title('Value of a_t across positions')
        ax.legend()

        # Right: a_t colored by observation
        ax = axes[1]
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
        for obs_val in range(min(L, 5)):
            mask = observations == obs_val
            positions = np.where(mask)[0]
            if len(positions) > 0:
                ax.scatter(positions, at_values[positions], s=8, alpha=0.5,
                          color=colors[obs_val % len(colors)], label=f'o_t={obs_val}')
        ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel('Position t')
        ax.set_ylabel('a_t')
        ax.set_title('a_t colored by observation value')
        ax.legend()

        plt.suptitle('State transition factor a_t (HMM experiment)')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved a_t plot to {save_path}")
    except Exception as e:
        print(f"Could not save plot: {e}")
