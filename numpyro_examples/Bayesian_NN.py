import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
import numpy as np
from numpyro.infer import MCMC, NUTS, Predictive

from vnn.bnn import BNN


# create artificial regression dataset
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

    rng_key = random.key(0)
    rng_key_infer, rng_key_pred = random.split(rng_key)

    N = 50
    D_X = 3
    X, Y, X_test = get_data(N=N, D_X=D_X)

    num_samples = 500
    model = BNN(dim_input=D_X, hidden_layer_sizes=(10, 10))
    sampler = NUTS(model)
    mcmc = MCMC(sampler, num_warmup=num_samples, num_samples=num_samples)
    mcmc.run(rng_key_infer, X, Y)
    posterior_samples = mcmc.get_samples()

    # predictions
    predictive = Predictive(model, posterior_samples=posterior_samples)
    predictions = predictive(rng_key_pred, X_test)["f"]

    # compute mean prediction and confidence interval around median
    mean_prediction = jnp.mean(predictions, axis=0)
    percentiles = np.percentile(predictions, [5.0, 95.0], axis=0)

    # make plots
    _, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)

    # plot training data
    ax.plot(X[:, 1], Y[:, 0], "kx")
    # plot 90% confidence level of predictions
    ax.fill_between(
        X_test[:, 1], percentiles[0, :, 0], percentiles[1, :, 0], color="lightblue"
    )
    # plot mean prediction
    ax.plot(X_test[:, 1], mean_prediction, "blue", ls="solid", lw=2.0)
    ax.set(xlabel="X", ylabel="Y", title="Expected regression function with 90% CI")
    plt.show()


if __name__ == "__main__":
    main()
