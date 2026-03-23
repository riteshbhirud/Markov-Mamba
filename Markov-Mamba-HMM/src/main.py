"""
Main entry point for HMM experiments.

Trains MambaZero (Mamba with convolution, no gating, no ReLU activation, no MLP)
on random HMM data and compares to the forward algorithm optimal predictor.

Usage:
    python main.py --config_format hmm [options]

Experiments:
    A: 1-layer MambaZero, M=2, L=2, w=2
    B: 1-layer MambaZero, M=2, L=2, w=3
    C: 2-layer MambaZero, M=2, L=2, w=2
"""

import os
import sys
import copy
import inspect
import argparse
import torch

import config
from models.utils import get_model
from optim.base import train_hmm


def get_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--config_format', default='hmm', choices=config.registered_formats())
    args, rem_args = parser.parse_known_args()
    return config.parse_args_with_format(format=args.config_format, base_parser=parser, args=rem_args, namespace=args)


def get_exp_name(args):
    exp_name = f"hmm_M{args.hmm_M}_L{args.hmm_L}_w{args.d_conv}_layers{args.n_layer}"
    exp_name += f"_d{args.d_model}_ds{args.d_state}_lr{args.lr}_bs{args.batch_size}"
    if args.wandb_run_prefix != 'none':
        exp_name = args.wandb_run_prefix + '_' + exp_name
    exp_name += f"_seed={args.seed}"
    return exp_name


def main(args):
    # Set up
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_default_dtype(args.dtype)

    args.device = torch.device(args.device)
    device_type = "cuda" if "cuda" in str(args.device) else "cpu"
    if device_type == "cuda":
        torch.cuda.set_device(args.device)

    generator = torch.Generator(device=args.device)
    if args.seed != 0:
        generator.manual_seed(args.seed)
    else:
        generator.seed()

    # Print experiment info
    print("="*60)
    print("HMM Experiment")
    print("="*60)
    print(f"Hidden states (M): {args.hmm_M}")
    print(f"Observations (L):  {args.hmm_L}")
    print(f"Dirichlet beta:    {args.hmm_beta}")
    print(f"Conv window (w):   {args.d_conv}")
    print(f"Num layers:        {args.n_layer}")
    print(f"d_model:           {args.d_model}")
    print(f"d_state:           {args.d_state}")
    print(f"Sequence length:   {args.sequence_length}")
    print(f"Iterations:        {args.iterations}")
    print(f"Batch size:        {args.batch_size}")
    print(f"MLP:               {'Yes' if not args.no_mlp else 'No (MambaZero)'}")
    print(f"Gating:            {'Yes' if args.gate else 'No'}")
    print(f"Convolution:       {'Yes' if args.conv else 'No'}")
    print(f"Fix A=1:           {'Yes' if args.fix_A else 'No'}")
    print("="*60)

    # Build model
    model = get_model(args).to(args.device)

    group_specs = model.get_parameter_group_specs()
    param_name_mapping = {p_name: p for p_name, p in model.named_parameters()}
    optimized_params_cnt = 0
    for g in group_specs:
        params = []
        for p_name in g["params"]:
            params += [param_name_mapping[p_name]]
        g["params"] = params
        optimized_params_cnt += sum([p.numel() for p in g["params"]])

    print(f"Number of optimized parameters: {optimized_params_cnt}")

    # Optimizer
    if args.opt == 'adamw':
        use_fused = (device_type == 'cuda') and ('fused' in inspect.signature(torch.optim.AdamW).parameters)
        extra_args_opt = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(group_specs, lr=args.lr, betas=(args.beta1, args.beta2),
                                       weight_decay=args.weight_decay, **extra_args_opt)
    else:
        optimizer = torch.optim.SGD(group_specs, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    # Scheduler
    if args.scheduler != 'none':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer, max_lr=args.lr, total_steps=args.iterations,
            pct_start=args.warmup_percent, anneal_strategy=args.scheduler,
            cycle_momentum=False, div_factor=1e2, final_div_factor=.05
        )
    else:
        scheduler = None

    # Wandb
    if args.wandb:
        import wandb
        params_copy = copy.deepcopy(vars(args))
        del params_copy['device']
        wandb.init(project=args.wandb_project, name=get_exp_name(args), config=params_copy)

    # Experiment directory
    exp_name = get_exp_name(args)
    ckpt_path = os.path.join(args.results_base_folder, args.dataset, args.model, exp_name)
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    elif os.path.isfile(os.path.join(ckpt_path, "metrics.json")):
        print(f"Already found completed experiment '{ckpt_path}'. Skipping.")
        sys.exit(0)

    print(f"\nTraining model={args.model}\n{vars(args)}\n")

    # Train
    train_hmm(
        model=model,
        opt=optimizer,
        scheduler=scheduler,
        iterations=args.iterations,
        acc_steps=args.acc_steps,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        M=args.hmm_M,
        L=args.hmm_L,
        beta=args.hmm_beta,
        generator=generator,
        eval_freq=args.eval_freq,
        ckpt_path=os.path.join(ckpt_path, "ckpt.pt"),
        extra_args=args,
    )

    # Save final model
    torch.save(model.state_dict(), os.path.join(ckpt_path, 'model_final.pt'))
    print(f"All results saved to {ckpt_path}")


if __name__ == "__main__":
    args = get_args()
    main(args)
