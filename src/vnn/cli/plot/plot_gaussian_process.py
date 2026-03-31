if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    from vnn.plot import update_mpl_params

    update_mpl_params()

    def f(x: np.ndarray) -> np.ndarray:
        return np.sin(x)

    def rbf_kernel(
        x: np.ndarray, y: np.ndarray | None = None, *, scale: float = 1.0
    ) -> np.ndarray:
        if y is None:
            y = x

        m = x.shape[0]
        n = y.shape[0]
        K = np.zeros(shape=(m, n))

        for i in range(m):
            for j in range(n):
                K[i, j] = np.exp(-(np.square(x[i] - y[j])) / (2 * scale**2))
        return K

    # ------------
    # Data
    # ------------
    N = 200
    scale = 0.1
    x = np.linspace(-5, 5, N)
    y = f(x) + scale * np.random.rand(*x.shape)

    # ------------
    # Prior
    # ------------
    n_prior_samples = 4
    M_0 = np.zeros_like(x)
    P_0 = rbf_kernel(x)
    std = np.sqrt(np.diag(P_0))
    prior_samples = np.random.multivariate_normal(M_0, P_0, size=(n_prior_samples,))

    plt.figure()
    plt.fill_between(
        x,
        M_0 - 1.96 * std,
        M_0 + 1.96 * std,
        alpha=0.2,
        label="±1.96 Std. dev.",
        color="gray",
    )
    for i in range(n_prior_samples):
        plt.plot(x, prior_samples[i, :], lw=1)

    plt.xlabel(r"$\mathbf{x}$")
    plt.ylabel(r"$f(\mathbf{x})$")
    plt.tick_params(direction="in", top=True, right=True)
    plt.legend(fancybox=False, edgecolor="black", loc="lower left")

    plt.tight_layout()
    plt.savefig("figures/gp_prior.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    # ------------
    # Posterior
    # ------------
    # Observe a few samples.
    obs_index = [10, 50, 100, 150, 190]
    n_obs = len(obs_index)
    x_obs = x[obs_index]
    y_obs = y[obs_index]
    Y_obs = y_obs.reshape(-1, 1)
    x_s = x

    K = rbf_kernel(x_obs)
    K_s = rbf_kernel(x_s, x_obs)
    I = scale ** 2 * np.eye(n_obs)
    inv = np.linalg.inv(K + I)
    K_ss = rbf_kernel(x_s, x_s)

    M_n = (K_s @ inv @ Y_obs).flatten()
    P_n = K_ss - K_s @ inv @ K_s.T
    std = np.sqrt(np.diag(P_n))

    n_posterior_samples = 4
    posterior_samples = np.random.multivariate_normal(
        M_n, P_n, size=(n_posterior_samples,)
    )

    plt.figure()

    plt.scatter(x_obs, y_obs, s=50, marker="x", c="black")

    plt.fill_between(
        x,
        M_n - 1.96 * std,
        M_n + 1.96 * std,
        alpha=0.2,
        label="±1.96 Std. dev.",
        color="gray",
    )
    for i in range(n_posterior_samples):
        plt.plot(x, posterior_samples[i, :], lw=1)

    plt.xlabel(r"$\mathbf{x}$")
    plt.ylabel(r"$f(\mathbf{x})$")
    plt.tick_params(direction="in", top=True, right=True)
    plt.legend(fancybox=False, edgecolor="black", loc="lower left")
    plt.tight_layout()
    plt.savefig("figures/gp_post.pdf", dpi=300, bbox_inches="tight")
    plt.close()
