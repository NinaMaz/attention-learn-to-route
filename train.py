import os
import time
from tqdm import tqdm
import torch
import math
import wandb

from torch.utils.data import DataLoader
from torch.nn import DataParallel
import torch.distributed as dist
import torch.multiprocessing as mp

from nets.attention_model import set_decode_type
from utils.log_utils import log_values
from utils import move_to, get_subroutes
from eval import get_best


def get_inner_model(model):
    return model.module if isinstance(model, DataParallel) else model


def validate(division_model, model, dataset, opts):
    # Validate
    print("Validating...")
    cost = rollout(division_model, model, dataset, opts)
    avg_cost = cost.mean()
    print(
        "Validation overall avg_cost: {} +- {}".format(
            avg_cost, torch.std(cost) / math.sqrt(len(cost))
        )
    )

    return avg_cost


def rollout(model, tsp_model, dataset, opts):
    # Put in greedy evaluation mode!

    set_decode_type(model, "greedy")
    model.eval()

    def eval_model_bat(bat):
        with torch.no_grad():
            bat = move_to(bat, opts.device)
            cost, log_likelihood, pi = model(bat, return_pi=True)
            subroutes = get_subroutes(pi, opts.device, pad_size=opts.graph_size)
            bat_with_depot = torch.cat([bat["depot"].unsqueeze(1), bat["loc"]], 1)
            l = []
            for n, i in enumerate(subroutes):
                temp = []
                for j in i:
                    temp.append(
                        bat_with_depot[n].gather(
                            0, j.unsqueeze(-1).expand_as(bat_with_depot[n])
                        )
                    )
                l.append(temp)
            cost = train_batch(
                tsp_model, l, opts
            )
            cost = torch.tensor(cost).unsqueeze(-1).to(log_likelihood)
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


def clip_grad_norms(param_groups, max_norm=math.inf):
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    :param optimizer:
    :param max_norm:
    :param gradient_norms_log:
    :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group["params"],
            max_norm
            if max_norm > 0
            else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2,
        )
        for group in param_groups
    ]
    grad_norms_clipped = (
        [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    )
    return grad_norms, grad_norms_clipped


def train_epoch(
    division_model,
    model,
    optimizer,
    baseline,
    lr_scheduler,
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

    # Put model in train mode!
    division_model.train()
    set_decode_type(division_model, "sampling")

    for batch_id, batch in enumerate(
        tqdm(training_dataloader, disable=opts.no_progress_bar)
    ):

        subroutes, bl_val, x, log_likelihood = division_model_forward(
            division_model,
            optimizer,
            baseline,
            epoch,
            batch_id,
            step,
            batch,
            tb_logger,
            opts,
        )

       
        cost = train_batch(
            model, subroutes, opts, step
        )

        cost = torch.tensor(cost).unsqueeze(-1).to(log_likelihood)
        # Calculate loss
        bl_val, bl_loss = baseline.eval(x, cost) if bl_val is None else (bl_val, 0)
        reinforce_loss = ((cost - bl_val) * log_likelihood).mean()
        loss = reinforce_loss + bl_loss

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
                tb_logger,
                opts,
            )

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
                "model": get_inner_model(division_model).state_dict(),
                "optimizer": optimizer.state_dict(),
                "rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all(),
                "baseline": baseline.state_dict(),
            },
            os.path.join(opts.save_dir, "epoch-{}.pt".format(epoch)),
        )
    # fix it here
    avg_reward = validate(division_model, model, val_dataset, opts)

    wandb.log({"val_avg_reward": avg_reward}, step=step)
    if not opts.no_tensorboard:
        tb_logger.log_value("val_avg_reward", avg_reward, step)

    baseline.epoch_callback(division_model, epoch)

    # lr_scheduler should be called at end of epoch
    lr_scheduler.step()


def train_batch(
    model, subroutes, opts, step = None
):
    model.eval()
    model.set_decode_type("greedy")
    results = []
    new_subroutes = []
    n_s = []
    for b in subroutes:
        new_subroutes.append(torch.stack(b, dim=0))
        n_s.append(len(b))
    if step:    
        wandb.log({"avg_len_subroutes": np.mean(n_s)}, step=step)
    new_subroutes = torch.cat(new_subroutes, dim = 0)
    tsp_dataloader = DataLoader(
        new_subroutes, batch_size=opts.batch_size
    )
    
    l_costs = []
    for b in tsp_dataloader:
        costs, log_likelihood = model(b)
        l_costs.append(costs)

    costs = torch.cat(l_costs, dim = 0)
    
    ind_p = 0
    for n in n_s:
        results.append(costs[ind_p:ind_p+n].sum(0))

    return(results)


def division_model_forward(
    model, optimizer, baseline, epoch, batch_id, step, batch, tb_logger, opts
):
    x, bl_val = baseline.unwrap_batch(batch)
    x = move_to(x, opts.device)
    bl_val = move_to(bl_val, opts.device) if bl_val is not None else None

    # Evaluate model, get costs and log probabilities
    cost, log_likelihood, pi = model(x, return_pi=True)

    subroutes = get_subroutes(pi, opts.device, pad_size=opts.graph_size)

    x_with_depot = torch.cat([x["depot"].unsqueeze(1), x["loc"]], 1)
    l = []
    for n, i in enumerate(subroutes):
        temp = []
        for j in i:
            temp.append(
                x_with_depot[n].gather(0, j.unsqueeze(-1).expand_as(x_with_depot[n]))
            )
        l.append(temp)

    return l, bl_val, x, log_likelihood
