#!/usr/bin/env python

import os
import json
import pprint as pp
import wandb

import torch
import torch.optim as optim

from nets.critic_network import CriticNetwork
from train import validate, get_inner_model, train_epoch
from reinforce_baselines import (
    NoBaseline,
    ExponentialBaseline,
    CriticBaseline,
    RolloutBaseline,
    WarmupBaseline,
)
from nets.attention_model import AttentionModel
from nets.pointer_network import PointerNetwork, CriticNetworkLSTM
from utils import torch_load_cpu, load_problem
from utils.replay_buffer import ReplayBuffer
from nets.knapsack_agent import KnapsackModelAC
from algorithms import AC, PPO

__spec__ = None  # for tracing with pdb


def run(opts):

    wandb.init(name=opts.run_name, project="SbRLCO",
       settings= {
           "_disable_stats": True,
           "system_sample_seconds": 999999999,
           # "disabled": True
       })
    # Pretty print the run args
    pp.pprint(vars(opts))

    # Set the random seed
    torch.manual_seed(opts.seed)

    # Optionally configure tensorboard
    tb_logger = None
    if not opts.no_tensorboard:
        from tensorboard_logger import Logger as TbLogger

        tb_logger = TbLogger(
            os.path.join(
                opts.log_dir,
                "{}_{}".format(opts.problem, opts.graph_size),
                opts.run_name,
            )
        )

    os.makedirs(opts.save_dir)
    # Save arguments so exact configuration can always be found
    with open(os.path.join(opts.save_dir, "args.json"), "w") as f:
        json.dump(vars(opts), f, indent=True)

    # Set the device
    device = opts.device if opts.device is not None else "cpu"
    opts.device = torch.device(device if opts.use_cuda else "cpu")
    print(opts.device)


    # Figure out what's the problem
    problem = load_problem(opts.problem)

    # Load data from load_path
    load_data = {}
    assert (
        opts.load_path is None or opts.resume is None
    ), "Only one of load path and resume can be given"
    load_path = opts.load_path if opts.load_path is not None else opts.resume
    if load_path is not None:
        print("  [*] Loading data from {}".format(load_path))
        load_data = torch_load_cpu(load_path)
    knpsck_model = KnapsackModelAC(opts.embedding_dim, n_encode_layers=opts.n_encode_layers,
                                   normalization=opts.normalization, encoder_cls=opts.knapsack_enc).to(opts.device)
    if problem.NAME == 'cvrp':
        buffer = ReplayBuffer(
            opts.buffer_size,
            (opts.graph_size+1, opts.embedding_dim),
            (opts.graph_size+1, opts.embedding_dim),
            opts.device,
        )
    else:
        buffer = ReplayBuffer(
            opts.buffer_size,
            (opts.graph_size, opts.embedding_dim),
            (opts.graph_size, opts.embedding_dim),
            opts.device,
        )
    print(torch.cuda.device_count())
    # Initialize model
    model_class = {
        "attention": AttentionModel,
        "pointer": PointerNetwork,
    }.get(opts.model, None)
    assert model_class is not None, "Unknown model: {}".format(model_class)
    model = model_class(
        opts.embedding_dim,
        opts.hidden_dim,
        problem,
        n_encode_layers=opts.n_encode_layers,
        mask_inner=True,
        mask_logits=True,
        normalization=opts.normalization,
        tanh_clipping=opts.tanh_clipping,
        checkpoint_encoder=opts.checkpoint_encoder,
        shrink_size=opts.shrink_size,
        buffer=buffer,
        graph_size=opts.graph_size,
    ).to(opts.device)

    # if opts.use_cuda and torch.cuda.device_count() > 1:
    #     model = torch.nn.DataParallel(model)

    # Overwrite model parameters by parameters to load
    model_ = get_inner_model(model)
    model_.load_state_dict({**model_.state_dict(), **load_data.get("model", {})})
    wandb.watch(model)
    # Initialize baseline
    if opts.baseline == "exponential":
        baseline = ExponentialBaseline(opts.exp_beta)
    elif opts.baseline == "critic" or opts.baseline == "critic_lstm":
        assert problem.NAME == "tsp", "Critic only supported for TSP"
        baseline = CriticBaseline(
            (
                CriticNetworkLSTM(
                    2,
                    opts.embedding_dim,
                    opts.hidden_dim,
                    opts.n_encode_layers,
                    opts.tanh_clipping,
                )
                if opts.baseline == "critic_lstm"
                else CriticNetwork(
                    2,
                    opts.embedding_dim,
                    opts.hidden_dim,
                    opts.n_encode_layers,
                    opts.normalization,
                )
            ).to(opts.device)
        )
    elif opts.baseline == "rollout":
        baseline = RolloutBaseline(model, problem, opts)
    else:
        assert opts.baseline is None, "Unknown baseline: {}".format(opts.baseline)
        baseline = NoBaseline()

    if opts.bl_warmup_epochs > 0:
        baseline = WarmupBaseline(
            baseline, opts.bl_warmup_epochs, warmup_exp_beta=opts.exp_beta
        )

    # Load baseline from data, make sure script is called with same type of baseline
    if "baseline" in load_data:
        baseline.load_state_dict(load_data["baseline"])

    # Initialize optimizer
    optimizer = optim.Adam(
        [{"params": model.parameters(), "lr": opts.lr_model}]
        + (
            [{"params": baseline.get_learnable_parameters(), "lr": opts.lr_critic}]
            if len(baseline.get_learnable_parameters()) > 0
            else []
        )
    )
    knpsck_optimizer = optim.Adam(
        [{"params": knpsck_model.parameters(), "lr": opts.lr_knapsack}]
    )

    # Load optimizer state
    if "optimizer" in load_data:
        optimizer.load_state_dict(load_data["optimizer"])
        for state in optimizer.state.values():
            for k, v in state.items():
                # if isinstance(v, torch.Tensor):
                if torch.is_tensor(v):
                    state[k] = v.to(opts.device)

    # Initialize learning rate scheduler, decay by lr_decay once per epoch!
    lr_scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, lambda epoch: opts.lr_decay**epoch
    )
    knpsck_lr_scheduler = optim.lr_scheduler.LambdaLR(
        knpsck_optimizer, lambda epoch: opts.lr_decay**epoch
    )

    # Start the actual training loop
    val_dataset = problem.make_dataset(
        size=opts.graph_size,
        num_samples=opts.val_size,
        filename=opts.val_dataset,
        distribution=opts.data_distribution,
    )

    if opts.resume:
        epoch_resume = int(
            os.path.splitext(os.path.split(opts.resume)[-1])[0].split("-")[1]
        )

        torch.set_rng_state(load_data["rng_state"])
        if opts.use_cuda:
            torch.cuda.set_rng_state_all(load_data["cuda_rng_state"])
        # Set the random states
        # Dumping of state was done before epoch callback, so do that now (model
        #  is loaded)
        baseline.epoch_callback(model, epoch_resume)
        print("Resuming after {}".format(epoch_resume))
        opts.epoch_start = epoch_resume + 1

    avg_reward = validate(model, knpsck_model, val_dataset, opts)
    wandb.log({"val_avg_reward": avg_reward}, step=0)

    if opts.knapsack_alg == "ac":
        knpsck_alg = AC(knpsck_model, knpsck_optimizer, opts.loss_weights, opts.max_grad_norm,
                        0.99, opts.symmetric_force)
    elif opts.knapsack_alg == "ppo":
        knpsck_alg = PPO(knpsck_model, knpsck_optimizer, opts.loss_weights, opts.max_grad_norm,
                        0.99, opts.symmetric_force, 0.1,
                        0.2, 10, torch.ones(1, 1, dtype=torch.bool, device=opts.device))
    else:
        raise NotImplementedError("Currently supported algorithms: 'ac', 'ppo'")

    if not opts.eval_only:
        for epoch in range(opts.epoch_start, opts.epoch_start + opts.n_epochs):
            train_epoch(
                model,
                knpsck_alg,
                optimizer,
                baseline,
                lr_scheduler,
                knpsck_lr_scheduler,
                epoch,
                val_dataset,
                problem,
                tb_logger,
                opts,
            )


if __name__ == "__main__":
    from options import get_options

    run(get_options())
