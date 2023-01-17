import torch
from torch import Tensor
from torch.nn import functional as F


def logp_nonzero(z: Tensor, dim: int = -1) -> Tensor:
    r"""`\log P(\xi \neq 0)` with `P(\xi_j = 1) = \sigma(z_j)` assuming independence"""

    # P(\xi \neq 0) = 1 - P(\xi = 0) = 1 - \prod_j P(\xi_j = 0)
    #               = 1 - \prod_j \sigma(-z_j)
    #               = 1 - \exp\{\sum_j \log \sigma(-z_j)\}
    #               = 1 - \exp\{- \sum_j \log (1 + e^{z_j})\}
    # softplus(z): z \mapsto \log(1 + e^z)
    return torch.log(-torch.expm1(-F.softplus(z).sum(dim)))


def sample_bernoulli(z: Tensor, *, dtype: torch.dtype = torch.float) -> Tensor:
    """Draw a sample from a 0-1 Bernoulli rv with =1 probability logit `z`"""
    u = torch.empty_like(z).uniform_()

    # threshold is a logistic(1) r.v.
    return z.ge(u.logit_()).to(dtype=dtype)


def sample_nonzero(
    z: Tensor, dim: int = -1, *, dtype: torch.dtype = torch.float
) -> tuple[Tensor, Tensor]:
    """Darw a non-zero boolean vector from the logits `z` with rejection sampling"""
    sample = torch.zeros_like(z, dtype=dtype)
    to_draw = torch.logical_not(sample.any(dim, keepdim=True))
    n_draws = torch.zeros_like(to_draw, dtype=torch.long)
    while to_draw.any():
        # resample the remaining vectors
        remaining = to_draw.expand_as(sample)
        n_draws[to_draw] += 1
        sample[remaining] = sample_bernoulli(z[remaining], dtype=dtype)

        to_draw = torch.logical_not(sample.any(dim, keepdim=True))

    return sample, n_draws


if __name__ == '__main__':
    z = torch.randn(4, 8) - 5

    print(z)
    print(logp_nonzero(z, -1).exp())
    print(sample_nonzero(z, -1))
