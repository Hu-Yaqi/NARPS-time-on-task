import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

df = pd.read_csv("runwise_parameters_fixed.csv")

plt.rcParams.update({
    "font.size": 20,
    "axes.titlesize": 22,
    "axes.labelsize": 21,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "axes.linewidth": 1.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

runs = [1, 2, 3, 4]

def plot_param(col, ylabel, title, color, outname, ylim=None):
    fig, ax = plt.subplots(figsize=(7.5, 5.8))

    for _, sub in df.groupby("subject"):
        sub = sub.sort_values("run")
        ax.plot(
            sub["run"], sub[col],
            color="0.45",
            alpha=0.45,
            linewidth=1.25,
            zorder=1
        )

    agg = df.groupby("run")[col].agg(["mean", "sem"])

    ax.errorbar(
        agg.index, agg["mean"], yerr=agg["sem"],
        marker="o",
        markersize=8,
        linewidth=3.2,
        elinewidth=2.4,
        capsize=7,
        capthick=2.4,
        color=color,
        zorder=3
    )

    ax.set_xlabel("Run")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=14)
    ax.set_xticks(runs)

    if ylim is not None:
        ax.set_ylim(ylim)

    ax.grid(False)

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    fig.tight_layout()
    fig.savefig(outname, format="pdf", bbox_inches="tight")
    plt.close(fig)


plot_param(
    "lambda",
    r"$\lambda$ estimate",
    r"$\lambda$ loss aversion across runs",
    "#534AB7",
    "lambda_loss_aversion.pdf"
)

plot_param(
    "alpha",
    r"$\alpha$ estimate",
    r"$\alpha$ utility curvature across runs",
    "#1D9E75",
    "alpha_curvature.pdf",
    ylim=(0, 1.05)
)

plot_param(
    "beta",
    r"$\beta$ estimate",
    r"$\beta$ choice consistency across runs",
    "#D85A30",
    "beta_consistency.pdf"
)

with PdfPages("parameters_3pages.pdf") as pdf:
    for col, ylabel, title, color, ylim in [
        ("lambda", r"$\lambda$ estimate", r"$\lambda$ loss aversion across runs", "#534AB7", None),
        ("alpha", r"$\alpha$ estimate", r"$\alpha$ utility curvature across runs", "#1D9E75", (0, 1.05)),
        ("beta", r"$\beta$ estimate", r"$\beta$ choice consistency across runs", "#D85A30", None),
    ]:
        fig, ax = plt.subplots(figsize=(7.5, 5.8))

        for _, sub in df.groupby("subject"):
            sub = sub.sort_values("run")
            ax.plot(sub["run"], sub[col], color="0.45", alpha=0.45, linewidth=1.25, zorder=1)

        agg = df.groupby("run")[col].agg(["mean", "sem"])

        ax.errorbar(
            agg.index, agg["mean"], yerr=agg["sem"],
            marker="o", markersize=8, linewidth=3.2,
            elinewidth=2.4, capsize=7, capthick=2.4,
            color=color, zorder=3
        )

        ax.set_xlabel("Run")
        ax.set_ylabel(ylabel)
        ax.set_title(title, pad=14)
        ax.set_xticks(runs)

        if ylim is not None:
            ax.set_ylim(ylim)

        ax.grid(False)

        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

print("Saved:")
print("lambda_loss_aversion.pdf")
print("alpha_curvature.pdf")
print("beta_consistency.pdf")
print("parameters_3pages.pdf")
