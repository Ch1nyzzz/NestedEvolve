"""Circle Packing 五框架对比折线图 — best run per framework."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Aesthetic config ─────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Times New Roman", "DejaVu Serif"],
    "font.size":        11,
    "axes.labelsize":   12,
    "axes.titlesize":   13,
    "legend.fontsize":  10,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "axes.linewidth":   0.8,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "xtick.direction":  "in",
    "ytick.direction":  "in",
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "xtick.minor.size": 2.0,
    "ytick.minor.size": 2.0,
    "grid.linewidth":   0.5,
    "grid.alpha":       0.35,
    "legend.framealpha":0.92,
    "legend.edgecolor": "#cccccc",
    "legend.frameon":   True,
})

# ── Color palette ────────────────────────────────────────────────────────────
COLORS = {
    "Skill-Evolve (Ours)": "#e41a1c",   # red
    "AdaEvolve":           "#377eb8",   # blue
    "EvoX":                "#4daf4a",   # green
    "Base-Evolve (Ours)":  "#ff7f00",   # orange
    "OpenEvolve":          "#984ea3",   # purple
}
MARKERS = {
    "Skill-Evolve (Ours)": "o",
    "AdaEvolve":           "s",
    "EvoX":                "^",
    "Base-Evolve (Ours)":  "D",
    "OpenEvolve":          "v",
}
LSTYLES = {
    "Skill-Evolve (Ours)": "-",
    "AdaEvolve":           "-",
    "EvoX":                "--",
    "Base-Evolve (Ours)":  "-.",
    "OpenEvolve":          ":",
}

def running_best(scores):
    """Convert per-iteration scores to running best."""
    best = []
    cur = 0.0
    for s in scores:
        cur = max(cur, s)
        best.append(cur)
    return best

# ── Data: best run per framework ─────────────────────────────────────────────

# Skill-Evolve: run2 redo (best=0.9984)
skill_evolve = [
    0.925604,0.925604,0.925604,0.925604,0.925604,0.936383,0.941443,0.951469,
    0.993291,0.993291,0.993291,0.994657,0.994657,0.997499,0.997499,0.997499,
    0.997499,0.997499,0.997499,0.997499,0.997499,0.997499,0.997499,0.997499,
    0.997499,0.997499,0.997499,0.997499,0.997499,0.997499,0.997499,0.997499,
    0.997499,0.997499,0.997499,0.997499,0.997499,0.997499,0.997499,0.997499,
    0.997499,0.998373,0.998373,0.998373,0.998373,0.998373,0.998373,0.998373,
    0.998373,0.998373,
]

# Base-Evolve: run1 (best=0.9645, all 3 runs identical)
base_evolve = [
    0.8097,0.8572,0.8572,0.8572,0.9298,0.9298,0.9298,0.9298,
    0.9526,0.9566,0.9629,0.9645,0.9645,0.9645,0.9645,0.9645,
    0.9645,0.9645,0.9645,0.9645,0.9645,0.9645,0.9645,0.9645,
    0.9645,0.9645,0.9645,0.9645,0.9645,0.9645,0.9645,0.9645,
    0.9645,0.9645,0.9645,0.9645,0.9645,0.9645,0.9645,0.9645,
    0.9645,0.9645,0.9645,0.9645,0.9645,0.9645,0.9645,0.9645,
    0.9645,0.9645,
]

# OpenEvolve: run1 (best=0.9377) - per-iteration scores → running best
openevolve_raw = [
    0.5296,0.6203,0.7716,0.0000,0.7514,0.7108,0.5374,0.7569,0.2791,0.9377,
    0.7691,0.5506,0.9298,0.0000,0.2791,0.7554,0.5488,0.6290,0.9311,0.9298,
    0.2791,0.5629,0.2791,0.8121,0.5454,0.7461,0.6783,0.0000,0.2791,0.8121,
    0.5488,0.5576,0.7902,0.0000,0.2791,0.2791,0.8025,0.5772,0.0000,0.6569,
    0.2791,0.9377,0.8386,0.8121,0.8038,0.7823,0.0000,
]
openevolve = running_best(openevolve_raw)

# AdaEvolve: run1 (best=0.9981) - per-iteration scores → running best
adaevolve_raw = [
    0.7382,0.3947,0.7382,0.5617,0.7382,0.7382,0.5468,0.3947,0.7874,0.7874,
    0.7874,0.4440,0.9887,0.5450,0.2960,0.3700,0.0000,0.7953,0.0000,0.6046,
    0.8499,0.0000,0.9907,0.9348,0.0000,0.9905,0.8539,0.5496,0.8539,0.5594,
    0.9907,0.9943,0.5594,0.9907,0.9812,0.8539,0.9981,0.9971,0.7997,0.8200,
    0.0000,0.7224,0.7874,0.9537,0.4374,
]
adaevolve = running_best(adaevolve_raw)

# EvoX: run1 (best=0.9936) - per-iteration scores → running best
evox_raw = [
    0.8880,0.5296,0.8880,0.8880,0.8880,0.8880,0.8880,0.9936,0.5374,0.5496,
    0.8880,0.8880,0.8880,0.5374,0.8880,0.8880,0.8880,0.5296,0.9936,0.9936,
    0.8880,0.9961,0.3793,0.8298,0.9258,0.9374,0.7245,0.6907,0.5496,0.8464,
    0.8880,0.5504,0.7144,0.5375,0.3793,
]
evox = running_best(evox_raw)

# ── Pad all to 50 iterations ────────────────────────────────────────────────
def pad_to(lst, n=50):
    if len(lst) >= n:
        return lst[:n]
    return lst + [lst[-1]] * (n - len(lst))

data = {
    "Skill-Evolve (Ours)": pad_to(skill_evolve),
    "AdaEvolve":           pad_to(adaevolve),
    "EvoX":                pad_to(evox),
    "Base-Evolve (Ours)":  pad_to(base_evolve),
    "OpenEvolve":          pad_to(openevolve),
}

iters = np.arange(1, 51)

# ── Figure ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(1, 1, figsize=(7.0, 4.0))
fig.subplots_adjust(left=0.10, right=0.97, top=0.88, bottom=0.14)

ax.set_facecolor("#fafafa")
ax.yaxis.grid(True, linestyle="--", color="#aaaaaa", alpha=0.4, zorder=0)
ax.set_axisbelow(True)

# Plot order: lowest first so best lines are on top
plot_order = ["OpenEvolve", "Base-Evolve (Ours)", "EvoX", "AdaEvolve", "Skill-Evolve (Ours)"]

for name in plot_order:
    y = data[name]
    ax.plot(
        iters, y,
        color=COLORS[name],
        linestyle=LSTYLES[name],
        linewidth=1.8,
        marker=MARKERS[name],
        markersize=4.0,
        markevery=5,
        markeredgewidth=0.6,
        markeredgecolor="white",
        label=name,
        zorder=3,
    )

# Baseline line
ax.axhline(y=0.3642, color='gray', linestyle=':', linewidth=1.0, alpha=0.6, zorder=1)
ax.text(51, 0.3642, 'baseline', fontsize=8, color='gray', va='center')

# Target line
ax.axhline(y=1.0, color='gray', linestyle=':', linewidth=1.0, alpha=0.6, zorder=1)
ax.text(51, 1.0, 'target', fontsize=8, color='gray', va='center')

ax.set_xlabel("Iteration", labelpad=4)
ax.set_ylabel("Score (target_ratio)", labelpad=4)
ax.set_xlim(0.5, 52)
ax.set_ylim(0.3, 1.02)
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

# ── Legend ────────────────────────────────────────────────────────────────────
handles, labels = ax.get_legend_handles_labels()
# Reorder to match rank
order = [labels.index(n) for n in ["Skill-Evolve (Ours)", "AdaEvolve", "EvoX", "Base-Evolve (Ours)", "OpenEvolve"]]
ax.legend(
    [handles[i] for i in order],
    [labels[i] for i in order],
    loc="lower right",
    ncol=1,
    fontsize=9,
    frameon=True,
    edgecolor="#cccccc",
)

# ── Title ────────────────────────────────────────────────────────────────────
ax.set_title("Circle Packing (n=26)  —  Best Run per Framework  —  50 Iterations",
             fontsize=12, pad=8)

# ── Annotate final scores ────────────────────────────────────────────────────
annotations = [
    ("Skill-Evolve (Ours)", 0.9984, "#e41a1c"),
    ("AdaEvolve",           0.9981, "#377eb8"),
    ("EvoX",                0.9961, "#4daf4a"),
    ("Base-Evolve (Ours)",  0.9645, "#ff7f00"),
    ("OpenEvolve",          0.9377, "#984ea3"),
]
for name, score, color in annotations:
    ax.annotate(
        f"{score:.4f}",
        xy=(50, score), xytext=(-35, 8),
        textcoords="offset points",
        fontsize=8, color=color, fontweight="bold",
    )

# ── Save ─────────────────────────────────────────────────────────────────────
out = "/Users/erv1n/NestedAgent/meta_evolve/artifacts/circle_packing_comparison"
fig.savefig(out + ".pdf", dpi=300, bbox_inches="tight", format="pdf")
fig.savefig(out + ".png", dpi=300, bbox_inches="tight")
print(f"Saved to {out}.pdf and {out}.png")
