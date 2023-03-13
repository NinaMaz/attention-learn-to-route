import math
import os
import time
import torch
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from loss import ac_loss
from nets.attention_model import set_decode_type
from utils import move_to, get_subgraph
from utils.boolean_nonzero import logp_nonzero
from utils.log_utils import log_values
from train import train_batch, clip_grad_norms, validate, get_inner_model


def train_epoch_ac(
    model,
    knapsack_model,
    optimizer,
    knapsack_optimizer,
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
    wandb.log({"knpsck_learnrate_pg0": knapsack_optimizer.param_groups[0]["lr"]}, step=step)
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
    knapsack_model.train()    

    # Put model in train mode!

    set_decode_type(model, "sampling")
    for batch_id, batch in enumerate(
        tqdm(training_dataloader, disable=opts.no_progress_bar)
    ):
        
        logits_list = [] #
        cost_list = []
        select_list = []
        valid_list = []
        value_list = []
        x, bl_val = baseline.unwrap_batch(batch)
        # x = dataset item: {"loc", "demand", "depot"}
        # bl_val = baseline values [B]
        x = move_to(x, opts.device)
        ep_step = 0  #
        valid_mask = torch.ones(opts.batch_size, opts.graph_size, 1, dtype=torch.bool, device=opts.device)
        depot_mask = torch.ones(opts.batch_size, 1, 1, dtype=torch.bool, device=opts.device)
        while x["loc"].nonzero().nelement() !=0 and ep_step < opts.graph_size:
            bl_val = move_to(bl_val, opts.device) if bl_val is not None else None

            src_pad_mask = torch.cat([valid_mask, depot_mask], dim=1)
            logits, select_mask, value = knapsack_model(x, src_pad_mask.squeeze()) # mask: 1 = include, 0 = exclude
            subgraph, x = get_subgraph(x, select_mask)#
            cost = train_batch(
                model, optimizer, baseline, epoch, batch_id, step, subgraph, bl_val, tb_logger, opts #
            )
            ls = logits[:,1:,:]
            cost_list.append(cost)
            logits_list.append(ls)
            value_list.append(value)
            select_list.append(select_mask)
            valid_list.append(valid_mask.clone())
            valid_mask = valid_mask * torch.logical_not(select_mask)
            # masked_logits_list.append(torch.cat((torch.ones(logits.shape[0], 1, 1).to(logits), mask*logits[:,1:,:]), dim = 1)) #
            # reg_list.append((-1)*logp_nonzero(logits[:,1:,:], dim = 1))#
            # done_list.append(mask.)
            ep_step += 1 #
        data_logits = torch.stack(logits_list, dim = 0) # [L, Bs, Nn, 1]
        data_cost = torch.stack(cost_list, dim = 0) # [L, Bs]
        data_value = torch.stack(value_list, dim=0) # [L, Bs, 1]
        data_select = torch.stack(select_list, dim=0) # [L, Bs, Nn, 1]
        data_valid = torch.stack(valid_list, dim=0) # [L, Bs, Nn, 1]
        policy_loss, value_loss, entropy_loss = ac_loss(data_logits, data_value, data_select, data_valid, data_cost, opts.symmetric_force)
        knapsack_loss = policy_loss + value_loss + entropy_loss
#         print('REG ',reg_value.mean())
#         print('KNP ',(data_cost*data_mask.sum(2).squeeze()).mean())
        knapsack_optimizer.zero_grad()
        knapsack_loss.backward()
        # Clip gradient norms and get (clipped) gradient norms for logging
        grad_norms = clip_grad_norms(knapsack_optimizer.param_groups, opts.max_grad_norm)
        knapsack_optimizer.step()
        wandb.log({"knapsack_loss": knapsack_loss.item()}, step=step)
        wandb.log({"knapsack_avg_cost": data_cost.mean().item()}, step=step)
        wandb.log({"entropy_loss": entropy_loss.item()}, step=step)
        wandb.log({"value_loss": value_loss.item()}, step=step)
        wandb.log({"actor_loss": policy_loss.item()}, step=step)
        wandb.log({"ep_length": torch.any(data_valid, dim=-1).count_nonzero(dim=0).to(torch.float32).mean()}, step=step)
        wandb.log({"avg_action_prob": (torch.sigmoid(data_logits) * data_valid).sum() / data_valid.count_nonzero() }, step=step)
        wandb.log({"Step": step}, step=step)
        step += 1

    epoch_duration = time.time() - start_time
    print(
        "Finished epoch {}, took {} s".format(
            epoch, time.strftime("%H:%M:%S", time.gmtime(epoch_duration))
        )
    )

    if (
        opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0
    ) or epoch == opts.n_epochs - 1:
        print("Saving model and state...")
        torch.save(
            {
                "model": get_inner_model(model).state_dict(),
                "knapsack_model": get_inner_model(knapsack_model).state_dict(),
                "optimizer": optimizer.state_dict(),
                "rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all(),
                "baseline": baseline.state_dict(),
            },
            os.path.join(opts.save_dir, "epoch-{}.pt".format(epoch)),
        )

    avg_reward = validate(model, knapsack_model, val_dataset, opts)
    print(avg_reward)
    wandb.log({"val_avg_reward": avg_reward}, step=step)
    if not opts.no_tensorboard:
        tb_logger.log_value("val_avg_reward", avg_reward, step)

    baseline.epoch_callback(model, epoch)

    # lr_scheduler should be called at end of epoch
    lr_scheduler.step()
    knapsack_scheduler.step()

