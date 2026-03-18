import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np


def plot_masked_diffusion_trajectory(
    tokens,
    times=(0.0, 0.25, 0.75, 1.0),
    mask_token="[MASK]",
    seed=7,
    savepath="masked_diffusion_trajectory_aligned.pdf",
):
    """
    Visualize one monotone masked-diffusion trajectory.

    Each token gets a reveal time r_j ~ Uniform[0,1].
    At time t, token j is visible iff r_j <= t, else it is masked.

    This guarantees:
      - t=0   : fully masked
      - t=1   : fully revealed
      - rows are perfectly aligned because every token cell has the same size
    """
    rng = np.random.default_rng(seed)
    n = len(tokens)

    # Monotone reveal times
    reveal_times = rng.uniform(0.0, 1.0, size=n)

    sequences = []
    for t in times:
        seq_t = [tok if reveal_times[j] <= t else mask_token for j, tok in enumerate(tokens)]
        sequences.append(seq_t)

    # Fixed cell size for all tokens
    max_token_len = max(max(len(tok) for tok in tokens), len(mask_token))
    cell_w = max(1.4, 0.24 * max_token_len + 0.9)
    cell_h = 0.8
    x_gap = 0.18
    y_gap = 1.35

    left_margin = 1.8
    right_margin = 0.6
    top_margin = 0.8
    bottom_margin = 0.4

    total_w = left_margin + n * cell_w + (n - 1) * x_gap + right_margin
    total_h = top_margin + len(times) * y_gap + bottom_margin

    fig, ax = plt.subplots(figsize=(max(12, total_w * 0.9), total_h * 0.9))
    ax.set_xlim(0, total_w)
    ax.set_ylim(0, total_h)
    ax.axis("off")

    for row_idx, (t, seq) in enumerate(zip(times, sequences)):
        y = total_h - top_margin - (row_idx + 1) * y_gap

        # Time label
        ax.text(
            0.15,
            y + cell_h / 2,
            f"t = {t:g}",
            fontsize=14,
            ha="left",
            va="center",
            fontweight="bold",
        )

        for j, tok in enumerate(seq):
            x = left_margin + j * (cell_w + x_gap)
            is_mask = (tok == mask_token)

            patch = FancyBboxPatch(
                (x, y),
                cell_w,
                cell_h,
                boxstyle="round,pad=0.02,rounding_size=0.08",
                linewidth=1.2,
                edgecolor="#555555",
                facecolor="#f2f2f2" if is_mask else "#ffffff",
            )
            ax.add_patch(patch)

            ax.text(
                x + cell_w / 2,
                y + cell_h / 2,
                tok,
                fontsize=13,
                ha="center",
                va="center",
                family="monospace",
            )

    #ax.set_title("Masked Diffusion Language Model Trajectory", fontsize=16, pad=16)
    plt.tight_layout()
    plt.savefig(savepath, dpi=220, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    tokens = ["The", "cat", "sat", "on", "the", "mat", "."]
    plot_masked_diffusion_trajectory(tokens, seed=4)