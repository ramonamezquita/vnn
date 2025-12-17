## Prerequisites
Install [uv](https://docs.astral.sh/uv/) package manager.

## Installation
Clone the repository and sync with the `uv` virtual environment.
```
git clone git@github.com:ramonamezquita/vnn.git
cd vnn
uv sync
```

## Entrypoints
Entrypoints are located inside the `cli` directory and contain python scripts for training the available models. To ensure consistency with the projects dependencies, please run these scripts using `uv`. For example, the following command runs the MVE (Mean-Variance Estimation) training script and plots the results afterwards.
```
uv run src/vnn/mve.py --n_total_epochs 10000 --n_warmup_epochs 5000 --plot
```

Available scripts:
- `mve.py`: Minimizes negative likelihood function for both mean and variance simultaneously.



## Models

### Mean-Variance Estimator
Estimates the mean and the variance of the probability distribution of the target as a function of the input, given a Gaussian target error-distribution model.

**References**

[1] David A. Nix and Andreas S. Weigend. Estimating the mean and variance of the target probability
distribution. Proceedings of 1994 IEEE International Conference on Neural Networks (ICNN’94), 1:55–
60 vol.1, 1994

### More coming soon...


