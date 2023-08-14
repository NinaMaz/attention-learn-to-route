import torch
import torch.nn.functional as F

from algorithms.base import AlgBase
from algorithms.utils import A


class AC(AlgBase):
    def update(self, trajectory):
        x = trajectory["logits", "values", "actions", "done", "valid", "costs"]
        losses = AC.loss(*x, self.gamma)
        self.opt_step(losses)
        self.write_log(losses, trajectory)

    @staticmethod
    @torch.jit.script
    def loss(logits, values, actions, done, valid, costs, gamma: float = 0.99):
        # logits: [L, Bs, Nn], values: [L, Bs], actions: [L, Bs], done: [L, Bs], valid: [L, Bs, Nn], rewards: [L, Bs]
        not_done = done.logical_not()  # [L, Bs]
        num_not_done = torch.count_nonzero(not_done) + 1e-5
        # num_valid = torch.count_nonzero(valid) + 1e-5

        # compute qvalues based on rewards and predicted values of the next state
        advantage = A(values, costs, not_done, gamma)  # [L, Bs]

        log_probs = torch.log_softmax(logits, dim=-1)  # [L, Bs, Nn]
        log_prob_action = torch.gather(log_probs, -1, actions.unsqueeze(-1)).squeeze(-1)  # [L, Bs]
        probs = torch.softmax(logits, dim=-1)  # [L, Bs, Nn]

        # policy loss
        policy_loss = (log_prob_action * advantage.detach()).sum() / num_not_done
        # critic loss
        value_loss = advantage.pow(2).sum() / num_not_done
        # entropy loss
        entropy_loss = (log_probs * probs * valid).sum() / num_not_done

        return policy_loss, value_loss, entropy_loss