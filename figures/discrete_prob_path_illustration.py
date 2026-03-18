import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1) Continuous toy data sampler (adapted from your code)
# ============================================================

VISUALIZATION_BATCH_SIZE = 300000
RANGE_LIM = 5.0
GRID_SIZE = 21
TIMES = [0.0, 0.25, 0.5, 0.75, 1.0]
RNG_SEED = 0


def inf_train_gen_checkerboard(batch_size=200):
    """
    Continuous checkerboard terminal distribution, adapted from your code.
    Returns x_end only.
    """
    x1 = np.random.rand(batch_size) * 4 - 2
    x2_ = np.random.rand(batch_size) - np.random.randint(0, 2, batch_size) * 2
    x2 = x2_ + (np.floor(x1) % 2)
    x_end = np.concatenate([x1[:, None], x2[:, None]], axis=1) / 0.45
    return x_end


# ============================================================
# 2) Discretization helpers
# ============================================================

def make_grid(grid_size=41, range_lim=5.0):
    edges = np.linspace(-range_lim, range_lim, grid_size + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return edges, centers


def discretize_density_from_samples(samples, x_edges, y_edges):
    """
    Turn continuous samples into a discrete probability table on a grid.
    Output shape: (Nx, Ny), where axis 0 is x and axis 1 is y.
    """
    H, _, _ = np.histogram2d(
        samples[:, 0], samples[:, 1],
        bins=[x_edges, y_edges]
    )
    H = H.astype(np.float64)
    H /= H.sum()
    return H


def discretized_gaussian_1d(centers, sigma=1.0, mean=0.0):
    probs = np.exp(-0.5 * ((centers - mean) / sigma) ** 2)
    probs /= probs.sum()
    return probs


def sample_discrete_state(p, rng):
    flat_idx = rng.choice(p.size, p=p.ravel())
    ix, iy = np.unravel_index(flat_idx, p.shape)
    return ix, iy


# ============================================================
# 3) Factorized mixture path on a discrete grid
# ============================================================

def conditional_factorized_mixture_path(t, p0_x, p0_y, z_ix, z_iy, alpha=1.0):
    """
    p_t(x | z) for the factorized mixture path with kappa_t = t^alpha.
    Here z is a single target grid point indexed by (z_ix, z_iy).
    """
    kappa = t ** alpha
    onehot_x = np.zeros_like(p0_x)
    onehot_y = np.zeros_like(p0_y)
    onehot_x[z_ix] = 1.0
    onehot_y[z_iy] = 1.0

    px_t = (1.0 - kappa) * p0_x + kappa * onehot_x
    py_t = (1.0 - kappa) * p0_y + kappa * onehot_y
    return np.outer(px_t, py_t)


def marginal_factorized_mixture_path(t, p0_x, p0_y, p_data, alpha=1.0):
    """
    Exact marginal path induced by the factorized mixture path when:
      p0(x1, x2) = p0_x(x1) p0_y(x2)
      terminal distribution = p_data(x1, x2)
      kappa_t = t^alpha

    Using the expansion with kappa = t^alpha:
      p_t(x1,x2)
      = (1-kappa)^2 p0_x(x1)p0_y(x2)
        + kappa(1-kappa) p0_x(x1) p_data_y(x2)
        + kappa(1-kappa) p_data_x(x1) p0_y(x2)
        + kappa^2 p_data(x1,x2)
    """
    kappa = t ** alpha
    p_data_x = p_data.sum(axis=1)
    p_data_y = p_data.sum(axis=0)

    term_00 = (1.0 - kappa) ** 2 * np.outer(p0_x, p0_y)
    term_01 = kappa * (1.0 - kappa) * np.outer(p0_x, p_data_y)
    term_10 = kappa * (1.0 - kappa) * np.outer(p_data_x, p0_y)
    term_11 = kappa ** 2 * p_data
    return term_00 + term_01 + term_10 + term_11


# ============================================================
# 4) Plotting
# ============================================================

def plot_factorized_mixture_path_2d(
    grid_size=41,
    range_lim=5.0,
    batch_size=300000,
    times=(0.0, 0.25, 0.5, 0.75, 1.0),
    sigma0=1.0,
    rng_seed=0,
    savepath="factorized_mixture_path_2d.pdf",
    show_grid=False,
    alpha=1.0,
):
    rng = np.random.default_rng(rng_seed)
    np.random.seed(rng_seed)

    # Grid
    x_edges, x_centers = make_grid(grid_size, range_lim)
    y_edges, y_centers = make_grid(grid_size, range_lim)

    # Initial distribution: factorized discretized Gaussian
    p0_x = discretized_gaussian_1d(x_centers, sigma=sigma0, mean=0.0)
    p0_y = discretized_gaussian_1d(y_centers, sigma=sigma0, mean=0.0)

    # Terminal distribution: discretized checkerboard
    x_end = inf_train_gen_checkerboard(batch_size=batch_size)
    p_data = discretize_density_from_samples(x_end, x_edges, y_edges)

    # Choose one target point z from the checkerboard support for the top row
    z_ix, z_iy = sample_discrete_state(p_data, rng)
    z_x = x_centers[z_ix]
    z_y = y_centers[z_iy]

    # Build panels
    cond_panels = [conditional_factorized_mixture_path(t, p0_x, p0_y, z_ix, z_iy, alpha=alpha) for t in times]
    marg_panels = [marginal_factorized_mixture_path(t, p0_x, p0_y, p_data, alpha=alpha) for t in times]

    _fig, axs = plt.subplots(2, len(times), figsize=(3.2 * len(times), 6.2))

    for i, t in enumerate(times):
        # Top row: conditional path (per-subplot normalization)
        ax = axs[0, i]
        P = cond_panels[i]
        ax.imshow(
            P.T,
            origin="lower",
            extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
            interpolation="nearest",
            aspect="equal",
            vmin=0.0,
            vmax=P.max(),
        )
        ax.scatter([z_x], [z_y], marker="x", s=90, c="red", linewidths=2.0)
        ax.set_xticks([])
        ax.set_yticks([])
        if show_grid:
            ax.set_xticks(x_edges, minor=True)
            ax.set_yticks(y_edges, minor=True)
            ax.grid(which="minor", color="gray", linewidth=0.3, alpha=0.5)

        # Bottom row: marginal path (per-subplot normalization)
        ax = axs[1, i]
        P = marg_panels[i]
        ax.imshow(
            P.T,
            origin="lower",
            extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
            interpolation="nearest",
            aspect="equal",
            vmin=0.0,
            vmax=P.max(),
        )
        ax.set_xticks([])
        ax.set_yticks([])
        if show_grid:
            ax.set_xticks(x_edges, minor=True)
            ax.set_yticks(y_edges, minor=True)
            ax.grid(which="minor", color="gray", linewidth=0.3, alpha=0.5)

    for i, t in enumerate(times):
        axs[0, i].set_title(f"$t={t:.2f}$", fontsize=12)
    axs[0, 0].set_ylabel("Conditional prob. path", fontsize=12)
    axs[1, 0].set_ylabel("Marginal prob. path", fontsize=12)

    plt.tight_layout()

    # Save as PDF
    plt.savefig(savepath, format="pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    plot_factorized_mixture_path_2d(
        grid_size=GRID_SIZE,
        range_lim=RANGE_LIM,
        batch_size=VISUALIZATION_BATCH_SIZE,
        times=TIMES,
        sigma0=1.0,
        rng_seed=RNG_SEED,
        savepath="factorized_mixture_path_2d.pdf",
        show_grid=True,
        alpha=2.0,
    )