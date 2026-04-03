if __name__ == "__main__":
    import torch
    from pyro.infer import MCMC, NUTS
    from torch import nn

    from vnn.datasets import get_dataset
    from vnn.initializers import make_random_init
    from vnn.mlp import MLP
    from vnn.sampling import create_parser, get_pyro_distr, make_pyro_model

    parser = create_parser()
    args = parser.parse_args()
    print(f"Called `run_mh` with parameters: {vars(args)}")

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
    prior = get_pyro_distr(args.prior, args.prior_loc, args.prior_scale)
    module = MLP(
        n_features_in=1,
        hidden_layer_sizes=args.hidden_layer_sizes,
        n_features_out=1,
        hidden_activation_fn=nn.Tanh,
        weights_initializer=make_random_init(prior),
    )
    pyro_model = make_pyro_model(module, prior, sigma=3.0)

    # ============
    # MCMC
    # ============
    nuts_kernel = NUTS(pyro_model, target_accept_prob=0.7)
    mcmc = MCMC(nuts_kernel, num_samples=300)
    mcmc.run(xtorch, ytorch)

    # ============
    # Plot
    # ============
    if args.n_plot_samples > 0:
        import matplotlib.pyplot as plt
        from torch.func import functional_call

        samples = mcmc.get_samples(args.n_plot_samples)
        plt.figure(figsize=(8, 5))
        plt.scatter(x, y, alpha=0.4, label="data")

        for i in range(args.n_plot_samples):
            W = {k: v[i, ...] for k, v in samples.items()}
            mean = functional_call(module, W, xtorch).flatten().numpy()
            plt.plot(x, mean, alpha=1.0, color="blue")

        plt.ylabel("y")
        plt.show()
