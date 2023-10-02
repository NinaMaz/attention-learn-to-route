import wandb
import torch

from utils import clip_grad_norms


class AlgBase:
    def __init__(self, agent, opt, loss_weights, max_grad_norm, gamma):
        self.agent = agent
        self.opt = opt
        self.loss_weights = loss_weights
        self.max_grad_norm = max_grad_norm
        self.gamma = gamma
        self.step = 0

    def opt_step(self, losses):
        obj = sum(w * l for (w,l) in zip(self.loss_weights, losses))
        obj.backward()
        # Clip gradient norms and get (clipped) gradient norms for logging
        grad_norms = clip_grad_norms(self.opt.param_groups, self.max_grad_norm)
        self.opt.step()
        self.opt.zero_grad()

    def write_log(self, losses, trajectory, freq=100):
        if self.step % freq == 0:
            losses = [x.detach() for x in losses]
            valid, logits = trajectory["valid"].detach(), trajectory["logits"].detach()

            wandb.log({"total_loss": sum(losses).item()}, step=self.step)
            wandb.log({"actor_loss": losses[0].item()}, step=self.step)
            wandb.log({"value_loss": losses[1].item()}, step=self.step)
            wandb.log({"entropy_loss":losses[2].item()}, step=self.step)

            wandb.log({"ep_length": torch.any(valid, dim=-1).count_nonzero(dim=0).to(torch.float32).mean()},
                      step=self.step)
            # wandb.log({"avg_action_prob": (torch.sigmoid(logits) * valid).sum() / valid.count_nonzero()},
            #           step=self.step)
            wandb.log({"action_probs": torch.softmax(logits, dim=-1).mean(dim=[0,1]).cpu().tolist()}, step=self.step)
            wandb.log({"Step": self.step}, step=self.step)
        self.step += 1