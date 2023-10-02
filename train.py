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
from nets.agent_gnn import AgentGNN


def get_inner_model(model):
    return model.module if isinstance(model, DataParallel) else model


def validate(model, dataset, opts, problem):
    # Validate
    print("Validating...")
    cost = rollout(model, dataset, opts, problem)
    avg_cost = cost.mean()
    print(
        "Validation overall avg_cost: {} +- {}".format(
            avg_cost, torch.std(cost) / math.sqrt(len(cost))
        )
    )

    return avg_cost


def rollout(agent, dataset, opts, problem):
    agent.eval()
    def eval_model_bat(batch):
        with torch.no_grad():
            batch = move_to(batch, opts.device)
            state = problem.make_state(batch)
            cost, _ = agent.play(state, greedy=True)
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
        alg,
        baseline,
        lr_scheduler,
        epoch,
        val_dataset,
        problem,
        opts,
):
    print(
        "Start train epoch {}, lr={} for run {}".format(
            epoch, alg.opt.param_groups[0]["lr"], opts.run_name
        )
    )
    start_time = time.time()

    wandb.log({"learnrate": alg.opt.param_groups[0]["lr"]}, step=alg.step)

    # Generate new training data for each epoch
    training_dataset = baseline.wrap_dataset(
        problem.make_dataset(
            size=opts.graph_size,
            num_samples=opts.train_epoch_size,
            distribution=opts.data_distribution,
        )
    )
    training_dataloader = DataLoader(
        training_dataset, batch_size=opts.batch_size, num_workers=opts.num_workers
    )
    alg.agent.train()
    if isinstance(alg.agent.enc, AgentGNN) and hasattr(alg.agent.enc, "max_steps"):
        alg.agent.enc.num_steps = max(alg.agent.enc.min_steps,
                                      alg.agent.enc.max_steps * (epoch + 1) // opts.n_epochs)

    for batch_id, batch in enumerate(
            tqdm(training_dataloader, disable=opts.no_progress_bar)
    ):
        x, bl_val = baseline.unwrap_batch(batch)
        # x = dataset item: {"loc", "demand", "depot"}
        x = move_to(x, opts.device)
        state = problem.make_state(x)
        cost, traj = alg.agent.play(state)
        if bl_val is not None:
            bl_val = move_to(bl_val, opts.device)
            traj.append("b_rollout", bl_val)
        # assert cost.allclose(traj["costs"].sum(0))
        alg.update(traj)
        # print total gpu memory usage
        if alg.step % 100 == 0:
            wandb.log({"gpu_memory_allocated, GB": torch.cuda.memory_allocated(opts.device) / 1e9}, step=alg.step)
            wandb.log({"gpu_memory_reserved, GB": torch.cuda.memory_reserved(opts.device) / 1e9}, step=alg.step)
            wandb.log({"train_avg_cost": cost.mean()}, step=alg.step)
            wandb.log({"Example": opts.batch_size * alg.step}, step=alg.step)
            if isinstance(alg.agent.enc, AgentGNN):
                wandb.log({"AgentGNN steps": alg.agent.enc.num_steps}, step=alg.step)

    epoch_duration = time.time() - start_time
    print(
        "Finished epoch {}, took {} s".format(
            epoch, time.strftime("%H:%M:%S", time.gmtime(epoch_duration))
        )
    )
    avg_reward = validate(alg.agent, val_dataset, opts, problem)
    if (
            opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0
    ) or epoch == opts.n_epochs - 1:
        summary = wandb.run.summary.get("val_avg_reward")
        if summary is None or summary["min"] >= avg_reward:
            print("Saving model and state...")
            torch.save(
                {
                    "model": get_inner_model(alg.agent).state_dict(),
                    "optimizer": alg.opt.state_dict(),
                    "rng_state": torch.get_rng_state(),
                    "cuda_rng_state": torch.cuda.get_rng_state_all(),
                },
                os.path.join(opts.save_dir, "best_checkpoint.pt".format(epoch)),
            )

    wandb.log({"val_avg_reward": avg_reward}, step=alg.step)

    baseline.epoch_callback(alg.agent, epoch)
    # lr_scheduler should be called at end of epoch
    lr_scheduler.step()

