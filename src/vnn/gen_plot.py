import matplotlib.pyplot as plt

from vnn.ensemble import run

if __name__ == "__main__":
    n_estimators_grid = [1, 5, 10, 20]

    fig, axs = plt.subplots(nrows=2, ncols=2)
    axs = axs.flatten()

    for n, ax in zip(n_estimators_grid, axs):
        _, _, ax = run(
            n_estimators=n,
            ax=ax,
            plot=False,
            n_hidden_units=10,
            n_total_epochs=8000,
            n_warmup_epochs=4000,
        )

    plt.legend()
    plt.show()
