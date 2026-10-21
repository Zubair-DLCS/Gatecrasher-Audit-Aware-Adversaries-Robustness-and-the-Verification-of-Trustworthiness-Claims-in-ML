import json
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import FIG, RES

BLUE, ORANGE, AQUA, YELLOW, VIOLET, GREY = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                                            "#6250d6", "#8f8e89")
INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": "#c9c8c2", "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "text.color": INK, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10.5, "axes.titlesize": 11.5,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.grid": True,
    "grid.color": "#ebeae6", "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False})


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, name), dpi=170, bbox_inches="tight")
    plt.close(fig)


def _draws():
    return json.load(open(os.path.join(RES, "audit_draws.json")))


def fig1_bands():
    d = _draws()
    panels = [("fairness|logistic|demographic_parity", "Fairness audit\n(COMPAS, demographic parity gap)",
               "Gap between groups", BLUE, [100, 200, 500, 1000, 2000]),
              ("robustness|standard|0.1", "Robustness audit\n(Fashion-MNIST, PGD at 0.1)",
               "Accuracy under attack", ORANGE, [100, 200, 500, 1000]),
              ("privacy|unprotected|", "Privacy audit\n(canary membership test)",
               "Epsilon lower bound", VIOLET, [20, 50, 100, 200])]
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.2))
    for ax, (key, title, ylab, col, budgets) in zip(axes, panels):
        xs, lows, highs, means = [], [], [], []
        for i, b in enumerate(budgets):
            s = np.array(d[f"{key}|{b}" if not key.endswith("|") else f"{key}{b}"])
            lo, hi = np.percentile(s, [5, 95])
            ax.vlines(i, lo, hi, color=col, lw=9, alpha=0.30)
            ax.plot([i], [s.mean()], "o", color=col, ms=7)
            ax.annotate(f"{hi - lo:.2f}", (i, hi), xytext=(0, 5), textcoords="offset points",
                        ha="center", fontsize=8.5, color=INK2)
            xs.append(i); lows.append(lo); highs.append(hi); means.append(s.mean())
        span = max(highs) - min(lows)
        ax.set_ylim(min(lows) - 0.12 * span, max(highs) + 0.22 * span)
        ax.set_xticks(range(len(budgets)), [f"{b:,}" for b in budgets], fontsize=9)
        ax.set_xlabel("Queries" if "privacy" not in key else "Canaries inserted")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=10.5)
        ax.set_xlim(-0.6, len(budgets) - 0.4)
    fig.suptitle("The same model, audited 300 times: the shaded range is every claim threshold "
                 "the audit cannot settle", x=0.01, ha="left", fontweight="bold", fontsize=11.5)
    fig.tight_layout()
    save(fig, "fig1_indeterminate_bands.png")


def fig2_cost():
    s = pd.read_csv(os.path.join(RES, "stability.csv"))
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    series = [("fairness", "demographic_parity", "logistic", "Fairness (demographic parity)", BLUE, "o"),
              ("fairness", "equalised_odds", "logistic", "Fairness (equalised odds)", AQUA, "s"),
              ("robustness", None, "standard", "Robustness (PGD at 0.1)", ORANGE, "^")]
    for audit, metric, model, label, col, mk in series:
        g = s[(s.audit == audit) & (s.model == model)]
        if metric:
            g = g[g.metric == metric]
        g = g.sort_values("queries")
        ax.plot(g.queries, g.band_width, "-" + mk, color=col, lw=2, ms=7, label=label)
    ax.axhline(0.05, color=GREY, ls="--", lw=1.4)
    ax.text(105, 0.054, "5 percentage points", fontsize=9, color=INK2)
    ax.set_xscale("log")
    ax.set_xticks([100, 200, 500, 1000, 2000], ["100", "200", "500", "1,000", "2,000"])
    ax.minorticks_off()
    ax.set_xlabel("Queries the auditor is allowed")
    ax.set_ylabel("Width of the indeterminate band")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title("What a reproducible verdict costs")
    save(fig, "fig2_query_cost.png")


def fig3_strength():
    r = pd.read_csv(os.path.join(RES, "robustness_strength.csv"))
    r["strength"] = r.steps * r.restarts
    models = list(dict.fromkeys(r.model))
    fig, axes = plt.subplots(1, len(models), figsize=(5.6 * len(models), 4.3), sharey=True)
    eps_cols = {0.02: AQUA, 0.05: BLUE, 0.1: VIOLET, 0.15: ORANGE}
    for ax, m in zip(np.atleast_1d(axes), models):
        g = r[r.model == m]
        for eps, col in eps_cols.items():
            h = g[g.epsilon == eps].sort_values("strength")
            ax.plot(h.strength, h.robust_accuracy, "-o", color=col, lw=2, ms=5,
                    label=f"attack size {eps}")
        ax.set_xscale("log")
        ticks = sorted(g.strength.unique())
        ax.set_xticks(ticks, [str(t) for t in ticks], fontsize=8.5)
        ax.minorticks_off()
        ax.set_xlabel("Attack effort (steps x restarts)")
        ax.set_title(m.capitalize() + " model", fontsize=10.5)
        ax.set_ylim(0, 1)
    np.atleast_1d(axes)[0].set_ylabel("Reported robust accuracy")
    np.atleast_1d(axes)[0].legend(loc="lower left", fontsize=9)
    fig.suptitle('"Robust" is not a property of a model alone: the number an audit reports is '
                 "set by how far and how hard the auditor attacks",
                 x=0.01, ha="left", fontweight="bold", fontsize=11.5)
    fig.tight_layout()
    save(fig, "fig3_audit_strength.png")


def fig4_privacy():
    d = _draws()
    s = pd.read_csv(os.path.join(RES, "stability.csv"))
    p = s[s.audit == "privacy"]
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.3))
    ax = axes[0]
    for name, col in (("unprotected", ORANGE), ("dp-sgd", BLUE)):
        g = p[p.model == name].sort_values("canaries")
        ax.plot(g.canaries, g["mean"], "-o", color=col, lw=2, ms=6,
                label="no privacy protection" if name == "unprotected" else "trained with DP-SGD")
        ax.fill_between(g.canaries, g.band_lo, g.band_hi, color=col, alpha=0.18, lw=0)
    ax.axhline(1.0, color=GREY, ls="--", lw=1.4)
    ax.text(21, 1.06, "a claim of epsilon <= 1", fontsize=9, color=INK2)
    ax.set_xscale("log")
    ticks = [int(t) for t in sorted(p.canaries.unique())]
    ax.set_xticks(ticks, [str(t) for t in ticks]); ax.minorticks_off()
    ax.set_xlabel("Canaries the auditor may insert")
    ax.set_ylabel("Epsilon the audit can rule out")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title("What the audit can prove", fontsize=10.5)

    ax = axes[1]
    counts = [int(c) for c in sorted(p.canaries.unique())]
    rates = [float(np.mean(np.array(d[f"privacy|unprotected|{k}"]) <= 1.0)) for k in counts]
    ax.bar(range(len(counts)), rates, color=ORANGE, width=0.6)
    for i, v in enumerate(rates):
        ax.text(i, v + 0.02, f"{v:.0%}", ha="center", fontsize=10)
    ax.set_xticks(range(len(counts)), [str(c) for c in counts])
    ax.set_ylim(0, 1.1)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    ax.set_xlabel("Canaries the auditor may insert")
    ax.set_ylabel("Audits returning PASS")
    ax.set_title("How often a model with no privacy protection at all\npasses a claim of "
                 "epsilon <= 1", fontsize=10.5)
    fig.tight_layout()
    save(fig, "fig4_privacy_power.png")


def fig5_interference():
    a = pd.read_csv(os.path.join(RES, "interference_privacy.csv")).groupby(
        "noise_multiplier").agg(["mean", "std"])
    b = pd.read_csv(os.path.join(RES, "interference_robustness.csv")).groupby(
        "train_epsilon").agg(["mean", "std"])
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.3))
    ax = axes[0]
    x = np.arange(len(a.index))
    for col, label, c, mk in (("fairness_gap", "Fairness gap (COMPAS)", BLUE, "o"),
                              ("robust_accuracy", "Robust accuracy (Fashion)", ORANGE, "^"),
                              ("fashion_accuracy", "Clean accuracy (Fashion)", GREY, "s")):
        ax.errorbar(x, a[(col, "mean")], yerr=a[(col, "std")], fmt="-" + mk, color=c, lw=2,
                    ms=6, capsize=0, label=label)
    ax.set_xticks(x, [f"{v:g}" for v in a.index])
    ax.set_xlabel("DP-SGD noise multiplier (0 = no privacy protection)")
    ax.set_ylabel("Audited quantity")
    ax.legend(loc="center left", fontsize=9)
    ax.set_title("DP noise costs little accuracy and all robustness", fontsize=10.5)

    ax = axes[1]
    x = np.arange(len(b.index))
    ax.errorbar(x, b[("epsilon_lower_bound", "mean")], yerr=b[("epsilon_lower_bound", "std")],
                fmt="-o", color=VIOLET, lw=2, ms=6, capsize=0)
    ax.set_xticks(x, [f"{v:g}" for v in b.index])
    ax.set_xlabel("Adversarial training radius (0 = standard training)")
    ax.set_ylabel("Epsilon the audit can rule out")
    ax.set_title("The same privacy audit loses almost all of its power\non an adversarially "
                 "trained model", fontsize=10.5)
    fig.suptitle("A claim verified on one pipeline does not survive a change to another",
                 x=0.01, ha="left", fontweight="bold", fontsize=11.5)
    fig.tight_layout()
    save(fig, "fig5_interference.png")


ALL = [fig1_bands, fig2_cost, fig3_strength, fig4_privacy, fig5_interference]

if __name__ == "__main__":
    for f in ALL:
        try:
            f()
            print("ok", f.__name__)
        except (FileNotFoundError, KeyError) as e:
            print("skipped", f.__name__, e)
