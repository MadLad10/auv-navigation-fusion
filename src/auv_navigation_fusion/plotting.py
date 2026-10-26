"""Export plots and a replay animation; plotting stays an optional dependency."""
from pathlib import Path


def plot_showcase(showcase, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    import numpy as np

    output = Path(output)
    first = next(iter(showcase.values()))
    truth = np.asarray(first["truth_positions"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].plot(truth[:, 1], truth[:, 2], "k--", label="Ground truth")
    for name, result in showcase.items():
        p = np.asarray(result["predictions"])
        error = np.linalg.norm(p[:, 1:3] - truth[:, 1:3], axis=1)
        axes[0].plot(p[:, 1], p[:, 2], label=name)
        axes[1].plot(p[:, 0], error, label=name)
    axes[0].set(xlabel="World x (m)", ylabel="World y (m)", title="Position estimates")
    axes[0].axis("equal")
    axes[1].set(xlabel="Time (s)", ylabel="Horizontal error (m)", title="Unaligned error against truth")
    axes[1].axvline(10, color="0.4", linestyle=":", label="DVL lost")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.suptitle("Kinematic sensor replay · DVL dropout at 10 s · seed 0")
    fig.savefig(output / "trajectory.png", dpi=160)
    fig.savefig(output / "trajectory.svg")
    lines = axes[0].lines[1:]
    arrays = [np.asarray(result["predictions"]) for result in showcase.values()]

    def animate(index):
        end = int(index) + 1
        for line, array in zip(lines, arrays):
            line.set_data(array[:end, 1], array[:end, 2])
        axes[0].set_title(f"Position estimates · t={truth[index, 0]:.1f} s")
        return lines

    animation = FuncAnimation(fig, animate, frames=np.linspace(0, len(truth) - 1, 41, dtype=int), interval=125)
    animation.save(output / "dvl_dropout.gif", writer=PillowWriter(fps=8), dpi=90)
    plt.close(fig)
