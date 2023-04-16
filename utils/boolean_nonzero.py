import torch
from torch import Tensor
from torch.nn import functional as F


def sample_nonzero(
    z: Tensor, mask: Tensor, dim: int = -1, *, dtype: torch.dtype = torch.bool
) -> tuple[Tensor, Tensor]:
    """Darw a non-zero boolean vector from the logits `z` with rejection sampling"""
    sample = torch.zeros_like(z, dtype=dtype) # [Bs, Nn]
    probs = torch.sigmoid(z) * mask
    to_draw = mask.any(dim, keepdim=True) # don't need to sample when there is no valid actions
    n_draws = torch.zeros_like(to_draw, dtype=torch.long)
    N_TRIALS = 33
    while to_draw.any():
        # resample the remaining vectors
        remaining = to_draw.expand_as(sample)
        n_draws[to_draw] += 1
        if n_draws[to_draw][0] > N_TRIALS:
            ix0 = to_draw.squeeze().nonzero(as_tuple=True)[0]
            ix1 = torch.multinomial(probs[ix0], 1)[...,0]
            sample[ix0, ix1] = True
        else:
            sample += torch.bernoulli(probs).to(dtype=dtype) * remaining
        to_draw *= torch.logical_not(sample.any(dim, keepdim=True))
    return sample, n_draws


if __name__ == '__main__':
    z = torch.randn(4, 8) - 5

    print(z)
    print(sample_nonzero(z, -1))
