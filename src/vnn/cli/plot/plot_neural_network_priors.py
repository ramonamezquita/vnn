if __name__ == "__main__":
    from typing import Type

    import matplotlib.pyplot as plt
    import torch
    from torch.distributions import Cauchy, Distribution, Laplace, Normal

    from vnn.mlp import MLP
    from vnn.plot import update_mpl_params

    update_mpl_params()

    X = torch.linspace(-1, 1, 200).reshape(-1, 1)
    x = X.detach().numpy().flatten()

    distribution_types: list[Type[Distribution]] = [Laplace, Normal, Cauchy]
    n_samples: int = 5

    for type_ in distribution_types:
        fig, ax = plt.subplots(figsize=(5, 4))
        prior = type_(loc=0.0, scale=1.0)

        for _ in range(n_samples):
            mlp = MLP(
                n_features_in=1,
                hidden_layer_sizes=(80, 80, 100),
                hidden_activation_fn=torch.nn.Tanh,
                n_features_out=1,
                weights_distribution=prior,
            )

            y = mlp(X).detach().numpy().flatten()
            ax.plot(x, y, linewidth=1.5, alpha=0.9)

        ax.tick_params(direction="in", top=True, right=True)
        plt.tight_layout()
        plt.savefig(
            f"figures/w_init_{type_.__name__.capitalize()}.pdf",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
