import pandas as pd
import matplotlib.pyplot as plt

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

fig, ax = plt.subplots(figsize=(7.5, 5.8))

for _, sub in df.groupby("subject"):
    sub = sub.sort_values("run")
    ax.plot(
        sub["run"],
        sub["beta"],
        color="0.45",
        alpha=0.45,
        linewidth=1.25,
        zorder=1
    )

agg = df.groupby("run")["beta"].agg(["mean", "sem"])

ax.errorbar(
    agg.index,
    agg["mean"],
    yerr=agg["sem"],
    marker="o",
    markersize=6,
    linewidth=2.2,
    elinewidth=1.8,
    capsize=5,
    capthick=1.8,
    color="#D85A30",
    zorder=5
)

ax.set_xlabel("Run")
ax.set_ylabel(r"$\tau$ estimate")
ax.set_title(r"$\tau$ inverse temperature across runs", pad=14)
ax.set_xticks([1, 2, 3, 4])

ax.grid(False)

for spine in ax.spines.values():
    spine.set_linewidth(1.5)

fig.tight_layout()
fig.savefig("tau_inverse_temperature.pdf", format="pdf", bbox_inches="tight")
plt.close(fig)

print("Saved: tau_inverse_temperature.pdf")