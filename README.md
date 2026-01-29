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
Entrypoints are located inside the `src` directory and contain python scripts for training the available models. To ensure consistency with the projects dependencies, please run these scripts using `uv`. 

### `ensemble.py`

Train and evaluate a Deep Ensemble of mean–variance estimation (MVE) neural networks.


```
uv run src/vnn/ensemble.py --n_total_epochs 20000 --n_warmup_epochs 10000 --n_estimators 10 --n_jobs 4
```




