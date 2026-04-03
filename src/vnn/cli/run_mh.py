from argparse import Namespace

import torch
from torch import nn
from torch.distributions import Normal

from vnn.datasets import get_dataset
from vnn.initializers import make_random_init
from vnn.mlp import MLP
from vnn.sampling import (
    ProbabilisticModel,
    Proposal,
    create_parser,
    get_torch_distr,
    metropolis_hasting,
)


def main(args: Namespace) -> None:

    # ==========
    # Data
    # ==========
    ds = get_dataset(args.dataset)
    x, y = ds.sample(args.n_samples, seed=args.random_state)
    xtorch = torch.as_tensor(x, dtype=torch.float32).reshape(-1, 1)
    ytorch = torch.as_tensor(y, dtype=torch.float32).reshape(-1, 1)

    # ============
    # Prob Model
    # ============
    prior = get_torch_distr(args.prior, args.prior_loc, args.prior_scale)
    net = MLP(
        n_features_in=1,
        hidden_layer_sizes=args.hidden_layer_sizes,
        n_features_out=1,
        hidden_activation_fn=nn.Tanh,
        weights_initializer=make_random_init(prior),
    )
    model = ProbabilisticModel(
        nn=net, prior=prior, likelihood=lambda loc: Normal(loc, scale=ds.noise)
    )

    # ============
    # MCMC
    # ============
    proposal = Proposal(lambda x: Normal(loc=x, scale=args.step_size))
    initial_guess = {
        name: prior.expand(w.size()).sample()
        for name, w in model.get_named_parameters().items()
    }
    samples = metropolis_hasting(
        xtorch,
        ytorch,
        model=model,
        proposal=proposal,
        initial_guess=initial_guess,
        n_iterations=args.n_iterations,
    )

    # ============
    # Plot
    # ============
    if args.n_plot_samples > 0:
        import matplotlib.pyplot as plt

        samples = samples[-args.n_plot_samples :]
        plt.figure(figsize=(8, 5))
        plt.scatter(x, y, alpha=0.4, label="data")

        for W in samples:
            mean = model(W, xtorch).flatten().numpy()
            plt.plot(x, mean, alpha=1.0, color="blue")

        plt.ylabel("y")
        plt.show()


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    print(f"Called `run_mh` with parameters: {vars(args)}")
    main(args)
