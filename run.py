#!/usr/bin/env python

import os
import json
import pprint as pp
import wandb
import sys
from collections import defaultdict
from pathlib import Path
import time

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
from algorithms import AC, PPO, PG_Rollout
from nets.agent import Agent
import yaml

# torch.autograd.set_detect_anomaly(True)

__spec__ = None  # for tracing with pdb


def run(opts):
    run_name = f'{opts["problem"]}{opts["graph_size"]}_{opts["alg"]["type"]}_' \
               f'{opts["agent"]["encoder"]["cls"]}_{time.strftime("%Y-%m-%d-%H:%M:%S")}'
    opts["save_dir"] = str(Path(opts["save_dir"]) / run_name)
    opts["run_name"] = run_name
    wandb.init(
        name=run_name,
        project="SbRLCO",
        config=opts,
        settings= {
           "_disable_stats": True,
           "system_sample_seconds": 999999999,
           # "disabled": True
        })
    wandb.define_metric("val_avg_reward", summary="min")
    # Pretty print the run args
    # pp.pprint(vars(opts))
    print(yaml.dump(opts))
    opts = wandb.config

    # Set the random seed
    # torch.manual_seed(opts.seed)

    os.makedirs(opts.save_dir)
    # Save arguments so exact configuration can always be found
    with open(os.path.join(opts.save_dir, "args.json"), "w") as f:
        json.dump(dict(opts), f, indent=True)


    # Figure out what's the problem
    problem = load_problem(opts.problem)

    # Load data from load_path
    load_data = {}
    load_path = opts.get("load_path") or opts.get("resume")
    if load_path is not None:
        print("  [*] Loading data from {}".format(load_path))
        load_data = torch_load_cpu(load_path)

    agent = Agent(encoder_cls=opts.agent["encoder"]["cls"],
                  encoder_params=opts.agent["encoder"]["args"],
                  embedding_dim=opts.agent["embedding_dim"],
                  hidden_dim=opts.agent["hidden_dim"],
                  node_features_option=opts.agent["node_features_option"]).to(opts.device)
    print(torch.cuda.device_count())

    optimizer = optim.Adam([
        {"params": agent.enc.parameters(), "lr": opts.lr_encoder},
        {"params": agent.actor.parameters(), "lr": opts.lr_actor},
        {"params": agent.val_layers.parameters(), "lr": opts.lr_critic},
    ])
    # Initialize learning rate scheduler, decay by lr_decay once per epoch!
    lr_scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, lambda epoch: opts.lr_decay**epoch
    )

    baseline = NoBaseline()

    # Start the actual training loop
    val_dataset = problem.make_dataset(
        size=opts.graph_size,
        num_samples=opts.val_epoch_size,
        filename=opts.val_dataset,
        distribution=opts.data_distribution,
    )

    avg_reward = validate(agent, val_dataset, opts, problem)
    wandb.log({"val_avg_reward": avg_reward}, step=0)

    if opts.alg["type"] == "ac":
        alg = AC(agent, optimizer, **opts.alg["args"])
    elif opts.alg["type"] == "pg_rollout":
        baseline = RolloutBaseline(agent, problem, opts)
        alg = PG_Rollout(agent, optimizer, **opts.alg["args"])
    elif opts.alg["type"] == "ppo":
        alg = PPO(agent, optimizer, opts.loss_weights, opts.max_grad_norm,
                        0.99, opts.symmetric_force, 0.1,
                        opts.batch_size, 10, torch.ones(1, 1, dtype=torch.bool, device=opts.device))
    else:
        raise NotImplementedError("Currently supported algorithms: 'ac', 'ppo'")

    if not opts.eval_only:
        for epoch in range(opts.epoch_start, opts.epoch_start + opts.n_epochs):
            train_epoch(
                alg,
                baseline,
                lr_scheduler,
                epoch,
                val_dataset,
                problem,
                opts,
            )


if __name__ == "__main__":
    # load opts from yaml  (args)
    with open(sys.argv[1]) as f:
        opts = yaml.load(f, Loader=yaml.FullLoader)
    run(opts)
