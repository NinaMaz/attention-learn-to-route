import torch
import torch.nn.functional as F

from algorithms.base import AlgBase
from algorithms.utils import Q, Sampler


class PPO(AlgBase):
    def __init__(self, agent, opt, loss_weights, max_grad_norm, gamma, symmetric_force,
                 eps, batch_part, update_iters, depot_mask):
        super().__init__(agent, opt, loss_weights, max_grad_norm, gamma, symmetric_force)
        self.eps = eps
        self.batch_part = batch_part
        self.update_iters = update_iters
        self.depot_mask = depot_mask

    def update(self, trajectory):
        obs, logits, values, costs, actions, valid = trajectory[
            "obs", "logits", "values", "costs", "actions", "valid"]
        not_done = torch.any(valid, dim=-1)  # [L, Bs]
        qvalues = Q(values, costs, not_done, self.gamma)
        old_probs = torch.sigmoid(logits)
        minibatch_size = int(self.batch_part * logits.shape[0] * logits.shape[1])
        sampler = Sampler(minibatch_size,
                          [obs, actions, valid, qvalues, old_probs.detach()],
                          actions.shape[0] * actions.shape[1])

        for _ in range(self.update_iters):
            # perform gradient descent for several epochs
            batch_obs, batch_actions, batch_valid, batch_qvalues, batch_old_probs = sampler.get_next()
            src_pad_mask = torch.cat([
                self.depot_mask.expand(batch_valid.shape[0], -1), batch_valid], dim=1)
            batch_logits, _, batch_values = self.agent.forward(batch_obs, src_pad_mask)
            batch_logits, batch_values = batch_logits[:,1:].view(batch_old_probs.shape), batch_values.view(batch_qvalues.shape)
            losses = PPO.loss(batch_logits, batch_values, batch_actions,
                              batch_valid, batch_qvalues, batch_old_probs, eps=self.eps)
            self.opt_step(losses)
        self.write_log(losses, trajectory)

    @staticmethod
    @torch.jit.script
    def loss(logits, values, actions, valid, qvalues, old_probs, eps: float, symmetric_force: bool = True):
        # logits: [N, Nn], values: [N],
        # actions: [N, Nn], valid: [N, Nn], qvalues: [N], old_probs: [N, Nn]
        not_done = torch.any(valid, dim=-1)  # [N]
        num_not_done = torch.count_nonzero(not_done) + 1e-5
        num_valid = torch.count_nonzero(valid) + 1e-5

        # compute qvalues based on rewards and predicted values of the next state
        advantage = (qvalues - values) * not_done  # [N]

        log_probs = F.logsigmoid(logits)  # [N, Nn]
        probs = torch.sigmoid(logits) # [N, Nn]

        # policy grad loss
        if symmetric_force:
            force = actions.to(torch.float32) - actions.logical_not().to(torch.float32)
        else:
            force = actions

        # critic loss
        value_loss = advantage.pow(2).sum() / num_not_done
        # entropy loss
        entropy_loss = - (
                F.binary_cross_entropy_with_logits(logits, probs, reduction='none') * valid
        ).sum() / num_valid
        # policy loss
        adv = advantage.unsqueeze(-1).detach()
        ratio = probs / old_probs * valid * force # [N, Nn]
        obj = ratio * adv
        obj2 = torch.clamp(ratio, 1 - eps, 1 + eps) * adv
        clip_loss = torch.min(obj, obj2).sum() / num_valid

        return clip_loss, value_loss, entropy_loss