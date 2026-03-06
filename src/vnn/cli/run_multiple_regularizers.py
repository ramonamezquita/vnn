import argparse
import copy
import multiprocessing as mp
from typing import Any

import mlflow

from vnn.ensemble import create_parser, run
from vnn.mlflow_logging import log_to_mlflow

EXPERIMENT_NAME = "run_multiple_regularizers"
TRACKING_URI = "http://127.0.0.1:5000"


reg_configs = [
    # --- L1 ---
    {
        "reg_type": "l1",
        "l1_penalty": 1.0,
        "l2_penalty": 0.0,
        "cauchy_scale": 0.0,
    },
    # --- L2 ---
    {
        "reg_type": "l2",
        "l1_penalty": 0.0,
        "l2_penalty": 0.5,
        "cauchy_scale": 0.0,
    },
    # --- Cauchy ---
    {
        "reg_type": "cauchy",
        "l1_penalty": 0.0,
        "l2_penalty": 0.0,
        "cauchy_scale": 1.0,
    },
]


def make_configs(parsed_args: argparse.Namespace) -> list[dict]:
    """Creates experiment configurations.

    Combines `reg_configs` (global) and parsed cli args to create
    multiple training configurations.
    """

    base = vars(parsed_args)
    configs = []
    for reg_cfg in reg_configs:
        reg_parts = "__".join(f"{k}={v}" for k, v in reg_cfg.items())
        new_config = copy.deepcopy(base)
        new_config.update(reg_cfg)
        new_config["random_state"] = parsed_args.random_state

        # Add run name.
        new_config["run_name"] = f"{reg_parts}__seed={parsed_args.random_state}"
        configs.append(new_config)

    return configs


def train_with_config(config: dict[str, Any]) -> None:
    """Train a single configuration and log results to MLflow.

    Designed to be called in a worker process via multiprocessing.Pool.
    See https://mlflow.org/docs/latest/ml/tracking/tracking-api/

    Parameters
    ----------
    config : dict[str, Any]
        Configuration for the current run.
    """
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=config["run_name"]):
        run_result = run(
            dataset=config["dataset"],
            n_estimators=config["n_estimators"],
            n_total_epochs=config["n_total_epochs"],
            n_warmup_epochs=config["n_warmup_epochs"],
            n_jobs=config["n_jobs"],
            n_samples=config["n_samples"],
            hidden_layer_sizes=config["hidden_layer_sizes"],
            test_size=config["test_size"],
            random_state=config["random_state"],
            verbose=config["verbose"],
            learning_rate=config["learning_rate"],
            l2_penalty=config["l2_penalty"],
            l1_penalty=config["l1_penalty"],
            cauchy_scale=config["cauchy_scale"],
            activation_fn=config["activation_fn"],
        )
        log_to_mlflow(run_result, config)


def main():
    parser = create_parser()
    parsed_args = parser.parse_args()
    all_configs = make_configs(parsed_args)

    with mp.Pool(processes=parsed_args.n_jobs) as pool:
        pool.map(train_with_config, all_configs)

    print(f"Completed {len(all_configs)} experiments.")


if __name__ == "__main__":
    main()
