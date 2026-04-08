from argparse import Namespace
from functools import partial

from pyro.infer import MCMC, NUTS

from vnn.datasets import get_x_y
from vnn.mlp import MLP
from vnn.sampling import create_parser
from vnn.sampling.pyro import PyroProbabilisticModel, pyro_distr


def gen_run_name(args: Namespace) -> str:
    parts = [
        f"ds={args.dataset}",
        f"prior={args.prior}",
        f"n_samples={args.n_samples}",
        f"n_warmup_steps={args.n_warmup_steps}",
        f"step_size={args.step_size}",
    ]
    return "__".join(parts)


def main(args: Namespace) -> None:

    # ==========
    # Data
    # ==========
    x, y = get_x_y(args.dataset, args.random_state)
    X = x.reshape(-1, 1)
    Y = y.reshape(-1, 1)

    # ============
    # Model
    # ============
    fward = MLP(1, 1, args.hidden_layer_sizes)
    prior = pyro_distr(args.prior, args.prior_loc, args.prior_scale)
    likel = partial(pyro_distr, "normal", scale=args.likelihood_scale)
    model = PyroProbabilisticModel(fward, prior, likel)

    # ============
    # MCMC
    # ============
    nuts = NUTS(model, step_size=args.step_size)
    mcmc = MCMC(nuts, args.n_samples, args.n_warmup_steps)
    mcmc.run(X, Y)

    if args.output_dir:
        import os
        import pickle

        run_name = gen_run_name(args)
        with open(
            os.path.join(args.output_dir, run_name + ".pkl"), "wb"
        ) as output_file:
            pickle.dump(mcmc, output_file)

    # ============
    # Plot
    # ============
    if args.n_plot_samples > 0:
        import matplotlib.pyplot as plt

        from vnn.plot import plot_pyro_chain

        plot_pyro_chain(mcmc, model.fward, X, Y, args.n_plot_samples)
        plt.show()


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    print(f"Called `run_mh` with parameters: {vars(args)}")
    main(args)
