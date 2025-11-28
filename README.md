## Prerequisites
Install [uv](https://docs.astral.sh/uv/) package manager.

## Installation
Clone and the repository and sync with the `uv` virtual environment.
```
git clone git@github.com:ramonamezquita/vnn.git
cd vnn
uv sync
```

## Entrypoints
Entrypoints are located inside the `cli` directory and contain python scripts for training the available models. To ensure consistency with the projects dependencies, please run these scripts using `uv`.
```
uv run src/vnn/cli/deep_ensemble.py --epochs 1000 --plot
```

