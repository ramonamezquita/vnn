from argparse import Namespace

from torch import nn
from torch.distributions import Normal

from vnn.datasets import get_x_y
from vnn.initializers import make_random_init
from vnn.mlp import MLP
from vnn.sampling import (
    Proposal,
    TorchProbabilisticModel,
    create_parser,
    metropolis_hasting,
    torch_distr,
)


def main(args: Namespace) -> None:

    # ==========
    # Data
    # ==========
    x, y = get_x_y(args.dataset, args.random_state)
    X = x.reshape(-1, 1)
    Y = y.reshape(-1, 1)

    # ============
    # Prob Model
    # ============
    prior = torch_distr(args.prior, args.prior_loc, args.prior_scale)
    net = MLP(
        n_features_in=1,
        hidden_layer_sizes=args.hidden_layer_sizes,
        n_features_out=1,
        hidden_activation_fn=nn.Tanh,
        weights_initializer=make_random_init(prior),
    )
    model = TorchProbabilisticModel(
        net,
        prior=prior,
        likelihood=lambda loc: Normal(loc, scale=args.likelihood_scale)
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
        X,
        Y,
        model=model,
        proposal=proposal,
        initial_guess=initial_guess,
        n_iterations=args.n_samples,
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
            mean = model(W, X).flatten().numpy()
            plt.plot(x, mean, alpha=1.0, color="blue")

        plt.ylabel("y")
        plt.show()


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    print(f"Called `run_mh` with parameters: {vars(args)}")
    main(args)
