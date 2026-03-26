if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import torch
    from torch import nn
    from torch.distributions import Normal

    from vnn.datasets import make_cubic
    from vnn.mlp import MLP
    from vnn.sampling import ProbabilisticModel, Proposal, metropolis_hasting

    n_samples = 100
    n_features_in = 1
    n_features_out = 1
    hidden_layers_sizes = (10,)

    # ----------------------------
    # Data
    # ----------------------------
    ds = make_cubic()
    X, y = ds.sample(n_samples, seed=0)
    X = torch.as_tensor(X, dtype=torch.float32).reshape(-1, 1)
    y = torch.as_tensor(y, dtype=torch.float32).reshape(-1, 1)

    # ----------------------------
    # Prior
    # ----------------------------
    prior = Normal(loc=0.0, scale=1.0)

    # ----------------------------
    # Probabilistic Model
    # ----------------------------
    forward = MLP(
        n_features_in=n_features_in,
        hidden_layer_sizes=hidden_layers_sizes,
        n_features_out=n_features_out,
        weights_distribution=prior,
        hidden_activation_fn=nn.Tanh,
    )
    model = ProbabilisticModel(
        forward=forward,
        prior=prior,
        likelihood=lambda loc: Normal(loc, scale=3.0),
    )

    # ----------------------------
    # Proposal
    # ----------------------------
    proposal = Proposal(lambda x: Normal(loc=x, scale=0.1))

    # ----------------------------
    # Run MCMC
    # ----------------------------
    initial_guess = {
        name: torch.zeros_like(w) for name, w in forward.named_parameters()
    }
    samples = metropolis_hasting(
        X,
        y,
        model=model,
        proposal=proposal,
        initial_guess=initial_guess,
        n_iterations=10_000,
    )

    # ----------------------------
    # Plot posterior predictive
    # ----------------------------
    # Use a smooth grid for plotting
    x_plot = torch.linspace(X.min(), X.max(), 200).reshape(-1, 1)

    # Last 50 samples
    last_samples = samples[-50:]

    plt.figure(figsize=(8, 5))
    plt.scatter(X.numpy(), y.numpy(), alpha=0.4, label="data")

    for weights in last_samples:
        with torch.no_grad():
            model.set_parameters(weights)
            y_pred = model(x_plot)
        plt.plot(x_plot.numpy(), y_pred.numpy(), alpha=0.2, color="blue")

    plt.ylabel("y")
    plt.show()
