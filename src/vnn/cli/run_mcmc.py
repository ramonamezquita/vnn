import argparse


def create_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description="Approximate neural network posterior distribution with metropolist-hasting sampling.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )

    parser.add_argument(
        "--random_state",
        type=int,
        help="Random seed.",
        default=0,
    )

    parser.add_argument(
        "--n_iterations",
        type=int,
        help="Number of epochs.",
        default=1000,
    )

    parser.add_argument(
        "--n_samples",
        default=100,
        type=int,
    )
    parser.add_argument(
        "--hidden_layer_sizes",
        default=(100,),
        nargs="*",
        type=int,
    )
    parser.add_argument(
        "--activation_fn",
        type=str,
        default="tanh",
    )

    return parser


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import torch
    from torch import nn
    from torch.distributions import Cauchy, Normal

    from vnn.datasets import get_dataset
    from vnn.mlp import MLP
    from vnn.sampling import ProbabilisticModel, Proposal, metropolis_hasting

    parser = create_parser()
    args = parser.parse_args()

    # ----------------------------
    # Data
    # ----------------------------

    ds = get_dataset(args.dataset)
    X, y = ds.sample(args.n_samples, seed=0)
    X = torch.as_tensor(X, dtype=torch.float32).reshape(-1, 1)
    y = torch.as_tensor(y, dtype=torch.float32).reshape(-1, 1)

    # ----------------------------
    # Prior
    # ----------------------------
    prior = Cauchy(loc=0.0, scale=1.0)

    # ----------------------------
    # Probabilistic Model
    # ----------------------------
    n_features_in = 1
    n_features_out = 1
    hidden_layers_sizes = (80, 80, 100)
    net = MLP(
        n_features_in=n_features_in,
        hidden_layer_sizes=args.hidden_layers_sizes,
        n_features_out=n_features_out,
        hidden_activation_fn=nn.Tanh,
        weights_distribution=prior,
    )
    model = ProbabilisticModel(
        nn=net, prior=prior, likelihood=lambda loc: Normal(loc, scale=0.05)
    )

    # ----------------------------
    # Proposal
    # ----------------------------
    proposal = Proposal(lambda x: Normal(loc=x, scale=0.001))

    # ----------------------------
    # Run MCMC
    # ----------------------------
    initial_guess = {
        name: torch.zeros_like(w) for name, w in model.get_named_parameters().items()
    }
    samples = metropolis_hasting(
        X,
        y,
        model=model,
        proposal=proposal,
        initial_guess=initial_guess,
        n_iterations=500_000,
    )

    # ----------------------------
    # Plot posterior predictive
    # ----------------------------
    x_plot = torch.linspace(X.min(), X.max(), 200).reshape(-1, 1)
    last_samples = samples[-1000:]

    plt.figure(figsize=(8, 5))
    plt.scatter(X.numpy(), y.numpy(), alpha=0.4, label="data")

    for weights in last_samples:
        with torch.no_grad():
            model.set_parameters(weights)
            y_pred = model(x_plot)
        plt.plot(x_plot.numpy(), y_pred.numpy(), alpha=0.2, color="blue")

    plt.ylabel("y")
    plt.show()
