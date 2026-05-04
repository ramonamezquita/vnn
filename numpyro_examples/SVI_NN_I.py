import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
import numpy as np
from numpyro.infer import SVI, Predictive, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
from numpyro.infer.initialization import init_to_sample
from numpyro.optim import ClippedAdam

from vnn.bnn import BNN


def get_data(N: int, D_X: int, sigma_obs: float = 0.05, N_test: int = 500):
    D_Y = 1  # create 1d outputs
    np.random.seed(0)
    X = jnp.linspace(-1, 1, N)
    X = jnp.power(X[:, np.newaxis], jnp.arange(D_X))
    W = 0.5 * np.random.randn(D_X)
    Y = jnp.dot(X, W) + 0.5 * jnp.power(0.5 + X[:, 1], 2.0) * jnp.sin(4.0 * X[:, 1])
    Y += sigma_obs * np.random.randn(N)
    Y = Y[:, np.newaxis]
    Y -= jnp.mean(Y)
    Y /= jnp.std(Y)

    assert X.shape == (N, D_X)
    assert Y.shape == (N, D_Y)

    X_test = jnp.linspace(-1.3, 1.3, N_test)
    X_test = jnp.power(X_test[:, np.newaxis], jnp.arange(D_X))

    return X, Y, X_test


def main():
    rng_key = random.PRNGKey(0)
    dat_key, vi_key, pred_key = random.split(rng_key, 3)

    N = 50
    D_X = 3
    X, Y, X_test = get_data(N=N, D_X=D_X)

    # Variational inference with mean-field normal guide
    num_steps = 100_000
    model = BNN(dim_input=D_X, hidden_layer_sizes=(100,), sigma_obs=0.05)
    adam = ClippedAdam(6e-4)
    elbo = Trace_ELBO()
    guide = AutoNormal(model, init_loc_fn=init_to_sample)
    svi = SVI(model, guide, adam, elbo)
    svi_result = svi.run(rng_key=vi_key, num_steps=num_steps, X=X, Y=Y)

    # Prediction
    num_samples = 5000
    predictive = Predictive(
        model, guide=guide, num_samples=num_samples, params=svi_result.params
    )
    predictions = predictive(pred_key, X_test)["f"]

    # compute mean prediction and confidence interval around median
    mean_prediction = jnp.mean(predictions, axis=0)
    percentiles = np.percentile(predictions, [5.0, 95.0], axis=0)

    # make plots
    _, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)

    ax.plot(X[:, 1], Y[:, 1], "kx")
    ax.fill_between(
        X_test[:, 1], percentiles[0, :, 0], percentiles[1, :, 0], color="lightblue"
    )

    ax.plot(X_test[:, 1], mean_prediction, "blue", ls="solid", lw=2.0)
    ax.set(xlabel="X", ylabel="Y", title="Expected regression function with 90% CI")
    plt.show()

    plt.show()


if __name__ == "__main__":
    main()
