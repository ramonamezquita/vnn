import matplotlib.pyplot as plt

from vnn.ensemble import run


def main():
    n_estimators_grid = [1, 5, 10, 20]
    _, axs = plt.subplots(nrows=2, ncols=2)
    plt.subplots_adjust(hspace=0.5)
    axs = axs.flatten()

    for n, ax in zip(n_estimators_grid, axs):
        run(
            n_estimators=n,
            ax=ax,
            plot=False,
            n_hidden_units=10,
            n_total_epochs=8000,
            n_warmup_epochs=4000,
            dataset="cubic"
        )

    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
