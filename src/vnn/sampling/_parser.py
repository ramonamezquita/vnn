import argparse


def create_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description="Approximate neural network posterior distribution with metropolist-hasting sampling.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset",
        default="cubic_poly",
    )

    parser.add_argument(
        "--random_state",
        type=int,
        help="Random seed.",
        default=0,
    )
    parser.add_argument(
        "--num_samples",
        default=5,
        type=int,
        help="The number of samples that need to be generated, excluding the samples discarded during the warmup phase.",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        help="Number of warmup iterations.",
        default=5,
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
    parser.add_argument(
        "--step_size",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--prior",
        type=str,
        default="normal",
    )
    parser.add_argument(
        "--prior_loc",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--prior_scale",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--n_plot_samples",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
    )
    parser.add_argument(
        "--likelihood_scale",
        type=float,
        default=1.0,
    )

    return parser
