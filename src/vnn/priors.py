import numpyro.distributions as dist

_NAME_TO_CLS = {"normal": dist.Normal, "laplace": dist.Laplace, "cauchy": dist.Cauchy}


def get_prior(prior: str, loc: float = 0.0, scale: float = 1.0) -> dist.Distribution:
    if prior not in _NAME_TO_CLS:
        raise ValueError()
    return _NAME_TO_CLS[prior](loc, scale)


def supported_priors() -> list[str]:
    return list(_NAME_TO_CLS)
