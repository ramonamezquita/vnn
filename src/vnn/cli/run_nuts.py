import os
from argparse import Namespace
from functools import partial

from vnn.datasets import get_x_y
from vnn.mlp import MLP
from vnn.sampling import create_parser, nuts
from vnn.sampling.pyro import PyroProbabilisticModel, pyro_distr


def _gen_output_filename(args: Namespace) -> str:
    parts = [
        f"ds={args.dataset}",
        f"prior={args.prior}",
        f"num_samples={args.num_samples}",
        f"warmup_steps={args.warmup_steps}",
        f"step_size={args.step_size}",
    ]
    return "__".join(parts)


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    print(f"Called `run_nuts` with parameters: {vars(args)}")

    # ==========
    # Data
    # ==========
    x, y = get_x_y(args.dataset, args.random_state)
    X = x.reshape(-1, 1)
    Y = y.reshape(-1, 1)

    # ============
    # Model
    fward = MLP(1, 1, args.hidden_layer_sizes)
    prior = pyro_distr(args.prior, args.prior_loc, args.prior_scale)
    likel = partial(pyro_distr, "normal", scale=args.likelihood_scale)
    model = PyroProbabilisticModel(fward, prior, likel)

    # ============
    # MCMC
    # ============
    mcmc = nuts(
        X,
        Y,
        model=model,
        step_size=args.step_size,
        num_samples=args.num_samples,
        warmup_steps=args.warmup_steps,
    )

    if args.output_dir:
        import pickle

        output_file = os.path.join(args.output_dir, _gen_output_filename(args))
        with open(output_file, "wb") as file:
            pickle.dump(mcmc, file)

    if args.n_plot_samples > 0:
        import matplotlib.pyplot as plt

        from vnn.plot import plot_mcmc

        plot_mcmc(mcmc, X, y, args.n_plot_samples)
        plt.show()
