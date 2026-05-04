import argparse


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deep ensemble.")
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--num_warmup_epochs", default=1000, type=int)
    parser.add_argument("--num_total_epochs", default=1000, type=int)
    parser.add_argument("--num_data", default=100, type=int)
    parser.add_argument("--hidden_sizes", nargs="*", default=[10], type=int)
    parser.add_argument("--seed", default=0, type=int)
    return parser


def main(args: argparse.Namespace) -> None:
    import matplotlib.pyplot as plt
    from torch.distributions import Cauchy

    from vnn.datasets import get_dataset
    from vnn.initializers import get_weights_initializer
    from vnn.mve import predict_ensemble, train_ensemble
    from vnn.regularizers import make_map_regularizer

    print(f"Called `run_ensemble` with args: {vars(args)}")

    # get data
    ds = get_dataset("1d_block")
    x, y = ds.sample(args.num_data, args.seed)
    X = x.reshape(-1, 1)

    # get prior
    loc = 0.0
    scale = 1.0
    reg_penalty = 1 / (args.num_data)
    prior_distr = Cauchy(loc=loc, scale=scale)
    regularizer = make_map_regularizer(prior_distr, reg_penalty)
    initializer = get_weights_initializer("cauchy", loc, scale)

    grad_max_norm = 5.0
    num_estimators = 12

    # train
    ensemble = train_ensemble(
        X,
        y,
        lr=args.lr,
        num_estimators=num_estimators,
        num_total_epochs=args.num_total_epochs,
        num_warmup_epochs=args.num_warmup_epochs,
        hidden_sizes=args.hidden_sizes,
        initializer=initializer,
        regularizer=regularizer,
        grad_max_norm=grad_max_norm,
        random_state=args.seed,
    )

    # predict
    stacked_output = predict_ensemble(ensemble, X)

    # plot individial mean samples
    means = stacked_output[:, :, 0]
    fig, axes = plt.subplots(4, 3, figsize=(10, 12), sharex=True)
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        ax.plot(x, means[i, :])
        ax.set_title(f"Estimator {i}")
    plt.show()


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
