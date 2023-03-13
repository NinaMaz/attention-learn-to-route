import torch
import torch.nn.functional as F


@torch.jit.script
def ac_loss(logits, values, select_mask, valid_mask, rewards, symmetric_force: bool = False, gamma: float = 0.99):
    # logits: [L, Bs, Nn, 1], values: [L, Bs, 1],
    # select_mask: [L, Bs, Nn, 1], valid_mask: [L, Bs, Nn], rewards: [L, Bs]
    ls, bs, nn = logits.shape[:3]
    logits, values = logits.view(ls, bs, nn), values.view(ls, bs)
    select_mask, valid_mask, rewards = select_mask.view(ls, bs, nn), valid_mask.view(ls, bs, nn), rewards.view(ls, bs)

    not_done = torch.any(valid_mask, dim=-1)  # [L, Bs]
    valid_num = torch.count_nonzero(valid_mask, dim=[0, 2]).view(1, -1, 1)  # [1, Bs, 1]

    # compute qvalues based on rewards and predicted values of the next state
    next_values = (values.detach() * not_done).roll(-1, 0)
    qvalues = rewards + gamma * next_values
    advantage = (qvalues - values) * not_done

    log_probs = F.logsigmoid(logits)
    probs = torch.sigmoid(logits)

    # policy grad loss
    if symmetric_force:
        force = select_mask.to(torch.float32) - select_mask.logical_not().to(torch.float32)
    else:
        force = select_mask
    policy_loss = ( log_probs * valid_mask * force
            * advantage.detach().unsqueeze(-1) / valid_num).sum(dim=(0, 2)).mean()
    # critic loss
    value_loss = (advantage.pow(2) / not_done.count_nonzero(dim=0)).sum(0).mean()
    # entropy loss
    entropy_loss = - (
            F.binary_cross_entropy_with_logits(logits, probs, reduction='none') * valid_mask
    ).sum() / torch.count_nonzero(valid_mask)

    # debug = {}
    # for n in ['logits', 'values', 'select_mask', 'valid_mask', 'rewards']:
    #     debug[n] = eval(n)
    #
    # debug["not_done"] = not_done
    # debug["qvalues"] = qvalues
    # debug["advantage"] = advantage
    # debug["log_probs"] = log_probs
    # debug["probs"] = probs
    # for n in ['policy_loss', 'value_loss', 'entropy_loss']:
    #     debug[n] = eval(n)
    # torch.save(debug, "debug.pt")

    return policy_loss, value_loss, entropy_loss
