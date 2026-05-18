import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# --------------------------------------------------
# Your data should be shaped like:
# loss_aversion:      subjects x runs, e.g. (108, 4)
# sensitivity:        subjects x runs
# consistency:        subjects x runs
# --------------------------------------------------

# Example:
# loss_aversion = np.array(...)
# sensitivity = np.array(...)
# consistency = np.array(...)

runs = np.array([1, 2, 3, 4])

plt.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 15,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def sem(x):
    return np.nanstd(x, axis=0, ddof=1) / np.sqrt(np.sum(~np.isnan(x), axis=0))


def plot_parameter(
    data,
    color,
    title,
    y_label,
    output_pdf,
    ylim=None
):
    mean_vals = np.nanmean(data, axis=0)
    sem_vals = sem(data)

    fig, ax = plt.subplots(figsize=(6.5, 5.2))

    # Individual subject trajectories: darker and clearer than before
    for row in data:
        ax.plot(
            runs,
            row,
            color="0.65",      # darker grey
            alpha=0.35,        # more visible
            linewidth=1.0,
            zorder=1
        )

    # Group mean with SEM
    ax.errorbar(
        runs,
        mean_vals,
        yerr=sem_vals,
        color=color,
        marker="o",
        markersize=6,
        linewidth=2.8,
        capsize=5,
        capthick=2,
        elinewidth=2,
        zorder=3
    )

    ax.set_title(title, pad=12)
    ax.set_xlabel("Run")
    ax.set_ylabel(y_label)

    ax.set_xticks(runs)

    if ylim is not None:
        ax.set_ylim(ylim)

    ax.tick_params(axis="both", width=1.5, length=6)

    for spine in ax.spines.values():
        spine.set_linewidth(1.3)

    ax.grid(False)

    fig.tight_layout()
    fig.savefig(output_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)


# Individual PDFs
plot_parameter(
    loss_aversion,
    color="#5B5FC7",
    title=r"$\lambda$ (loss aversion) across runs",
    y_label=r"$\lambda$ estimate",
    output_pdf="loss_aversion_trajectory.pdf",
    ylim=None
)

plot_parameter(
    sensitivity,
    color="#1B9E77",
    title=r"$\alpha$ (sensitivity) across runs",
    y_label=r"$\alpha$ estimate",
    output_pdf="sensitivity_trajectory.pdf",
    ylim=(0, 1.05)
)

plot_parameter(
    consistency,
    color="#E76F51",
    title=r"$\beta$ (consistency) across runs",
    y_label=r"$\beta$ estimate",
    output_pdf="consistency_trajectory.pdf",
    ylim=None
)


# Optional: combine the three into one multi-page PDF
with PdfPages("parameter_trajectories_separate_pages.pdf") as pdf:
    for data, color, title, y_label, ylim in [
        (loss_aversion, "#5B5FC7", r"$\lambda$ (loss aversion) across runs", r"$\lambda$ estimate", None),
        (sensitivity, "#1B9E77", r"$\alpha$ (sensitivity) across runs", r"$\alpha$ estimate", (0, 1.05)),
        (consistency, "#E76F51", r"$\beta$ (consistency) across runs", r"$\beta$ estimate", None),
    ]:
        fig, ax = plt.subplots(figsize=(6.5, 5.2))

        for row in data:
            ax.plot(
                runs,
                row,
                color="0.65",
                alpha=0.35,
                linewidth=1.0,
                zorder=1
            )

        mean_vals = np.nanmean(data, axis=0)
        sem_vals = sem(data)

        ax.errorbar(
            runs,
            mean_vals,
            yerr=sem_vals,
            color=color,
            marker="o",
            markersize=6,
            linewidth=2.8,
            capsize=5,
            capthick=2,
            elinewidth=2,
            zorder=3
        )

        ax.set_title(title, pad=12)
        ax.set_xlabel("Run")
        ax.set_ylabel(y_label)
        ax.set_xticks(runs)

        if ylim is not None:
            ax.set_ylim(ylim)

        ax.tick_params(axis="both", width=1.5, length=6)

        for spine in ax.spines.values():
            spine.set_linewidth(1.3)

        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)