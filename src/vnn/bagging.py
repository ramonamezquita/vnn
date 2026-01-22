from sklearn.ensemble import BaggingRegressor

from vnn.datasets import get_dataset
from vnn.main import RunParams
from vnn.mve import MVERegressor


def run(run_params: RunParams) -> None:
    base_regressor = MVERegressor(
        n_total_epochs=run_params.n_total_epochs,
        n_hidden_units=run_params.n_hidden_units,
        n_warmup_epochs=run_params.n_warmup_epochs,
        learning_rate=run_params.learning_rate,
        activation_fn=run_params.activation_fn,
    )

    bagging = BaggingRegressor(
        base_regressor,
        run_params.n_estimators,
        max_samples=run_params.max_samples,
        bootstrap=run_params.bootstrap,
        n_jobs=run_params.n_jobs,
        random_state=run_params.seed,
        verbose=run_params.verbose,
    )

    X, y = get_dataset(run_params.dataset)
    bagging.fit(X, y)
