import torch
import torch.nn.functional as F

from algorithms.base import AlgBase
from algorithms.utils import A


class AC(AlgBase):
    def update(self, trajectory):
        x = trajectory["logits", "values", "actions", "valid", "costs"]
        losses = AC.loss(*x, self.symmetric_force, self.gamma)
        self.opt_step(losses)
        self.write_log(losses, trajectory)

    @staticmethod
    @torch.jit.script
    def loss(logits, values, actions, valid, costs, symmetric_force: bool = True, gamma: float = 0.99):
        # logits: [L, Bs, Nn], values: [L, Bs],
        # select_mask: [L, Bs, Nn], valid_mask: [L, Bs, Nn], rewards: [L, Bs]
        not_done = torch.any(valid, dim=-1)  # [L, Bs]
        num_not_done = torch.count_nonzero(not_done) + 1e-5
        num_valid = torch.count_nonzero(valid) + 1e-5

        # compute qvalues based on rewards and predicted values of the next state
        advantage = A(values, costs, not_done, gamma)

        log_probs = F.logsigmoid(logits)
        probs = torch.sigmoid(logits)

        # policy grad loss
        if symmetric_force:
            force = actions.to(torch.float32) - actions.logical_not().to(torch.float32)
        else:
            force = actions

        # policy loss
        policy_loss = (log_probs * valid * force
                       * advantage.detach().unsqueeze(-1)).sum() / num_valid
        # critic loss
        value_loss = advantage.pow(2).sum() / num_not_done
        # entropy loss
        entropy_loss = - (
                F.binary_cross_entropy_with_logits(logits, probs, reduction='none') * valid
        ).sum() / num_valid

        return policy_loss, value_loss, entropy_loss