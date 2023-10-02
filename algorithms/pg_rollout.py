import torch
import torch.nn.functional as F

from algorithms.base import AlgBase
from algorithms.utils import A


class PG_Rollout(AlgBase):
    def update(self, trajectory):
        x = trajectory["logits", "actions", "done", "valid", "costs"]
        bl_val = trajectory["b_rollout"]
        losses = PG_Rollout.loss(*x, bl_val, self.gamma)
        self.opt_step(losses)
        self.write_log(losses, trajectory)

    @staticmethod
    @torch.jit.script
    def loss(logits, actions, done, valid, costs, b_rollout, gamma: float = 0.99):
        # logits: [L, Bs, Nn], values: [L, Bs], actions: [L, Bs], done: [L, Bs], valid: [L, Bs, Nn], costs: [L, Bs]
        b_rollout = b_rollout.expand(costs.shape[1]//b_rollout.shape[1], -1).reshape(1, -1)
        not_done = done.logical_not()  # [L, Bs]
        num_not_done = torch.count_nonzero(not_done) + 1e-5
        # num_valid = torch.count_nonzero(valid) + 1e-5

        # compute qvalues based on rewards and predicted values of the next state
        advantage = (costs.sum(0, keepdim=True) - b_rollout) # [1, Bs]

        log_probs = torch.log_softmax(logits, dim=-1)  # [L, Bs, Nn]
        log_prob_action = torch.gather(log_probs, -1, actions.unsqueeze(-1)).squeeze(-1)  # [L, Bs]
        probs = torch.softmax(logits, dim=-1)  # [L, Bs, Nn]

        # policy loss
        policy_loss = (log_prob_action * advantage.detach()).mean(1).sum()
        # value_loss
        value_loss = torch.tensor(0, device=policy_loss.device)
        # entropy loss
        entropy_loss =  (log_probs * probs * valid).mean(dim=[0,1]).sum()

        return policy_loss, value_loss, entropy_loss