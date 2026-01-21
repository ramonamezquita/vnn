from sklearn.ensemble import BaggingRegressor

from vnn.datasets import get_dataset
from vnn.mve import MVERegressor
import matplotlib.pyplot as plt

if __name__ == "__main__":
    regressor = BaggingRegressor(
        MVERegressor(n_total_epochs=10000, n_hidden_units=20), n_estimators=20, random_state=0, bootstrap=False,
        n_jobs=4, verbose=2
    )
    X, y = get_dataset("sinusoidal")
    regressor.fit(X, y)
    Y = regressor.predict(X)


    ax = plt.subplot()

    # Mean prediction.
    ax.plot(X[:, 0], Y[:, 0], label="Network output", c="black")

    # Actual observations.
    ax.scatter(X[:, 0], y[:, 0], label="Observations", s=2, c="black", alpha=0.1)

    ax.set_xlabel("X")
    ax.set_ylabel("y")
    plt.legend()
    plt.title("Deep Ensemble Regression")
    plt.show()

