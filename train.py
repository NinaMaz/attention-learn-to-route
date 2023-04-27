import os
import time
from tqdm import tqdm
import torch
import math
import wandb

from torch.utils.data import DataLoader
from torch.nn import DataParallel

from algorithms.utils import Trajectory
from nets.attention_model import set_decode_type
from utils.log_utils import log_values
from utils import move_to, get_subgraph, clip_grad_norms, ReplayBuffer


def get_inner_model(model):
    return model.module if isinstance(model, DataParallel) else model


def validate(model, knapsack_model, dataset, opts):
    # Validate
    print("Validating...")

    cost = rollout(model, dataset, opts, knapsack_model)
    avg_cost = cost.mean()
    print(
        "Validation overall avg_cost: {} +- {}".format(
            avg_cost, torch.std(cost) / math.sqrt(len(cost))
        )
    )

    return avg_cost


def rollout(model, dataset, opts, knapsack_model = None):
    # Put in greedy evaluation mode!
    set_decode_type(model, "greedy")
    model.eval()
    if knapsack_model:
        knapsack_model.eval()

    def eval_model_bat(bat):
        with torch.no_grad():
            if knapsack_model:
                ep_step = 0
                cost = 0
                valid_mask = torch.ones(bat["loc"].shape[0], opts.graph_size, dtype=torch.bool, device=opts.device)
                depot_mask = torch.ones(bat["loc"].shape[0], 1, dtype=torch.bool, device=opts.device)
                while bat["loc"].nonzero().nelement() !=0:
                    assert (ep_step < opts.graph_size)
                    src_pad_mask = torch.cat([depot_mask, valid_mask], dim=1)
                    logits, mask, *_ = knapsack_model(move_to(bat, opts.device), src_pad_mask)
                    valid_mask = valid_mask * torch.logical_not(mask)
                    subgraph, bat = get_subgraph(move_to(bat, opts.device), mask)
                    ep_step += 1
                    partial_cost, _, _, _ = model(move_to(subgraph, opts.device))
                    cost += partial_cost
            else:
                cost, _, _, _ = model(move_to(bat, opts.device))
        return cost.data.cpu()

    return torch.cat(
        [
            eval_model_bat(bat)
            for bat in tqdm(
                DataLoader(dataset, batch_size=opts.eval_batch_size),
                disable=opts.no_progress_bar,
            )
        ],
        0,
    )


def train_epoch(
        model,
        knapsack_alg,
        optimizer,
        baseline,
        lr_scheduler,
        knapsack_scheduler,
        epoch,
        val_dataset,
        problem,
        tb_logger,
        opts,
):
    print(
        "Start train epoch {}, lr={} for run {}".format(
            epoch, optimizer.param_groups[0]["lr"], opts.run_name
        )
    )
    step = epoch * (opts.epoch_size // opts.batch_size)
    start_time = time.time()

    wandb.log({"learnrate_pg0": optimizer.param_groups[0]["lr"]}, step=step)
    wandb.log({"knpsck_learnrate_pg0": knapsack_alg.opt.param_groups[0]["lr"]}, step=step)
    if not opts.no_tensorboard:
        tb_logger.log_value("learnrate_pg0", optimizer.param_groups[0]["lr"], step)

    # Generate new training data for each epoch
    training_dataset = baseline.wrap_dataset(
        problem.make_dataset(
            size=opts.graph_size,
            num_samples=opts.epoch_size,
            distribution=opts.data_distribution,
        )
    )
    training_dataloader = DataLoader(
        training_dataset, batch_size=opts.batch_size, num_workers=1
    )
    model.train()
    knapsack_alg.agent.train()

    # Put model in train mode!

    set_decode_type(model, "sampling")
    for batch_id, batch in enumerate(
            tqdm(training_dataloader, disable=opts.no_progress_bar)
    ):
        x, bl_val = baseline.unwrap_batch(batch)
        # x = dataset item: {"loc", "demand", "depot"}
        # bl_val = baseline values [B]
        x = move_to(x, opts.device)
        ep_step = 0  #
        valid_mask = torch.ones(opts.batch_size, opts.graph_size, dtype=torch.bool, device=opts.device)
        depot_mask = torch.ones(opts.batch_size, 1, dtype=torch.bool, device=opts.device)
        traj = Trajectory()
        while x["loc"].nonzero().nelement() != 0:
            assert (ep_step < opts.graph_size)
            bl_val = move_to(bl_val, opts.device) if bl_val is not None else None
            src_pad_mask = torch.cat([depot_mask, valid_mask], dim=1)
            logits, select_mask, value = knapsack_alg.agent(x, src_pad_mask)  # mask: 1 = include, 0 = exclude
            traj.append("obs", x)
            subgraph, x = get_subgraph(x, select_mask)  #
            if opts.lr_model > 0:
                cost = train_batch(
                    model, optimizer, baseline, epoch, batch_id, step, subgraph, bl_val, tb_logger, opts  #
                )
            else:
                model.eval()
                with torch.no_grad():
                    cost, _, _, _ = model(subgraph)
            traj.append("logits", logits[:, 1:])
            traj.append("costs", cost)
            traj.append("values", value)
            traj.append("actions", select_mask)
            traj.append("valid", valid_mask)

            valid_mask = valid_mask * torch.logical_not(select_mask)
            ep_step += 1  #

        knapsack_alg.update(traj)
        step += 1
        # print total gpu memory usage
        wandb.log({"gpu_memory_allocated, GB": torch.cuda.memory_allocated(opts.device) / 1e9}, step=step)
        wandb.log({"gpu_memory_reserved, GB": torch.cuda.memory_reserved(opts.device) / 1e9}, step=step)

    epoch_duration = time.time() - start_time
    print(
        "Finished epoch {}, took {} s".format(
            epoch, time.strftime("%H:%M:%S", time.gmtime(epoch_duration))
        )
    )
    avg_reward = validate(model, knapsack_alg.agent, val_dataset, opts)
    if (
            opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0
    ) or epoch == opts.n_epochs - 1:
        summary = wandb.run.summary.get("val_avg_reward")
        if summary is None or summary["min"] >= avg_reward:
            print("Saving model and state...")
            torch.save(
                {
                    "model": get_inner_model(model).state_dict(),
                    "knapsack_model": get_inner_model(knapsack_alg.agent).state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "rng_state": torch.get_rng_state(),
                    "cuda_rng_state": torch.cuda.get_rng_state_all(),
                    "baseline": baseline.state_dict(),
                },
                os.path.join(opts.save_dir, "best_checkpoint.pt".format(epoch)),
            )

    wandb.log({"val_avg_reward": avg_reward}, step=step)
    if not opts.no_tensorboard:
        tb_logger.log_value("val_avg_reward", avg_reward, step)

    baseline.epoch_callback(model, epoch)

    # lr_scheduler should be called at end of epoch
    lr_scheduler.step()
    knapsack_scheduler.step()


def train_batch(
    model, optimizer, baseline, epoch, batch_id, step, x, bl_val, tb_logger, opts
):
    #x, bl_val = baseline.unwrap_batch(batch)
    #x = move_to(x, opts.device)
    #bl_val = move_to(bl_val, opts.device) if bl_val is not None else None

    # Evaluate model, get costs and log probabilities
    cost, log_likelihood, label_pred, label_true = model(x)

    # Evaluate baseline, get baseline loss if any (only for critic)
    bl_val, bl_loss = baseline.eval(x, cost) if bl_val is None else (bl_val, 0)

    # Calculate loss
    reinforce_loss = ((cost - bl_val) * log_likelihood).mean()
    rcrl_loss = 0.0
    if label_true is not None:
        rcrl_loss = torch.mean((label_pred - label_true) ** 2)
    loss = reinforce_loss + bl_loss + rcrl_loss

    # Perform backward pass and optimization step
    optimizer.zero_grad()
    loss.backward()
    # Clip gradient norms and get (clipped) gradient norms for logging
    grad_norms = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)
    optimizer.step()

    # Logging
    if step % int(opts.log_step) == 0:
        log_values(
            cost,
            grad_norms,
            epoch,
            batch_id,
            step,
            log_likelihood,
            reinforce_loss,
            bl_loss,
            rcrl_loss,
            tb_logger,
            opts,
        )
    return cost
