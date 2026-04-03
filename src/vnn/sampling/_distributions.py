import torch.distributions as dist


def get_torch_distr(
    name: str, loc: float = 0.0, scale: float = 1.0
) -> dist.Distribution:
    name_to_distr = {
        "normal": dist.Normal,
        "cauchy": dist.Cauchy,
        "laplace": dist.Laplace,
    }
    if name not in name_to_distr:
        raise ValueError(f"Distribution {name} not supported.")

    return name_to_distr[name](loc, scale)
