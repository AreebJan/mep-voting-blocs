"""Report figures (static PNG, 300 dpi). Each group keeps the same colour in every figure
and both terms; group names are written on the plots so colour is never the only cue."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, INK2, GRID, AXIS = "#191919", "#666666", "#e7e7e7", "#c2c2c2"
GROUP_COLOURS = {
    "EPP": "#257de0", "SD": "#ec4647", "RENEW": "#f2a600", "GREEN_EFA": "#008a17",
    "GUE_NGL": "#ee7fac", "ECR": "#00b894", "ID": "#6a4ce2", "PFE": "#6a4ce2",
    "ESN": "#f47230", "NI": "#9c9c9c",
}
GROUP_NAMES = {"EPP": "EPP", "SD": "S&D", "RENEW": "Renew", "GREEN_EFA": "Greens/EFA",
               "GUE_NGL": "The Left", "ECR": "ECR", "ID": "ID", "PFE": "PfE",
               "ESN": "ESN", "NI": "Non-attached"}
LABEL_COLOURS = {"group": "#257de0", "country": "#f47230", "party": "#00b894"}
LABEL_NAMES = {"group": "European group", "country": "Country", "party": "National party"}

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9, "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.titlesize": 10, "axes.titlecolor": INK, "legend.frameon": False,
    "savefig.dpi": 300, "savefig.bbox": "tight",
})


def svd_maps(maps, path=None):
    """maps: {term: (coords, labels_df, explained)}"""
    fig, axes = plt.subplots(1, len(maps), figsize=(3.6 * len(maps), 3.4))
    for ax, (term, (c, lab, ex)) in zip(np.atleast_1d(axes), maps.items()):
        g = lab.group.to_numpy()
        order = ["NI"] + [x for x in GROUP_COLOURS if x != "NI"]
        for grp in order:
            m = g == grp
            if m.any():
                ax.scatter(c[m, 0], c[m, 1], s=9, color=GROUP_COLOURS[grp], alpha=0.75,
                           linewidths=0.3, edgecolors="white", label=GROUP_NAMES[grp])
        groups = list(np.unique(g))
        pos = np.array([[np.median(c[g == x, 0]), np.median(c[g == x, 1])] for x in groups])
        for grp, (x, y) in zip(groups, _repel(pos, c)):
            ax.annotate(GROUP_NAMES[grp], (x, y), fontsize=7.5, color=INK, ha="center",
                        va="center", weight="bold",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
        ax.margins(x=0.12, y=0.06)
        ax.set_title(f"Term {term}  ({ex[0]:.0%} + {ex[1]:.0%} of variance)", loc="left")
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.set_xticks([]); ax.set_yticks([])
    _save(fig, path)


def resolution_sweep(sweeps, path=None):
    """sweeps: {term: DataFrame from rq1.resolution_sweep}"""
    fig, axes = plt.subplots(1, len(sweeps), figsize=(3.6 * len(sweeps), 2.8), sharey=True)
    for ax, (term, s) in zip(np.atleast_1d(axes), sweeps.items()):
        ends = {col: s[f"AMI_{col}"].iloc[-1] for col in ["group", "party", "country"]}
        ys = _spread(list(ends.values()), gap=0.07)
        for (col, _), y in zip(ends.items(), ys):
            ax.plot(s.gamma, s[f"AMI_{col}"], color=LABEL_COLOURS[col], lw=2,
                    marker="o", ms=3.5, label=LABEL_NAMES[col])
            ax.annotate(LABEL_NAMES[col], (s.gamma.iloc[-1], y), xytext=(5, 0),
                        textcoords="offset points", va="center", fontsize=7.5, color=INK2,
                        annotation_clip=False)
        ax.axvline(1.0, color=AXIS, lw=0.8, ls="--")
        ax.set_xscale("log")
        ax.set_xticks([0.1, 0.3, 1, 3, 6]); ax.set_xticklabels(["0.1", "0.3", "1", "3", "6"])
        ax2 = ax.secondary_xaxis("top")
        ax2.set_xticks(s.gamma.iloc[::2]); ax2.set_xticklabels(s.n_communities.iloc[::2])
        ax2.tick_params(labelsize=7, colors=INK2, length=0)
        ax2.set_xlabel("communities found", fontsize=7.5, color=INK2)
        ax.set_title(f"Term {term}", loc="left")
        ax.set_xlabel("Resolution γ (log scale)")
        ax.set_ylim(0, 1)
        ax.set_xlim(0.08, 14)
    np.atleast_1d(axes)[0].set_ylabel("AMI with label (0 = chance)")
    _top_legend(fig, np.atleast_1d(axes)[0])
    _save(fig, path)


def policy_areas(tables, path=None):
    """tables: {term: DataFrame from rq1.policy_areas}; dot plot, AMI group vs country."""
    fig, axes = plt.subplots(1, len(tables), figsize=(3.6 * len(tables), 2.8), sharey=True)
    tables = {k: v for k, v in tables.items()}
    all_labels = sorted(set().union(*[set(t.label) for t in tables.values()]))
    ypos = {lbl: i for i, lbl in enumerate(all_labels)}
    for ax, (term, t) in zip(np.atleast_1d(axes), tables.items()):
        t = t[~t.skipped.astype(bool)]
        for col, mk in [("group", "o"), ("country", "s")]:
            ax.scatter(t[f"AMI_{col}"], t.label.map(ypos), color=LABEL_COLOURS[col], s=28,
                       marker=mk, label=LABEL_NAMES[col], zorder=3, edgecolors="white",
                       linewidths=0.6)
        for lbl in all_labels:
            if lbl not in set(t.label):
                ax.text(0.02, ypos[lbl], "too few votes (< 300)", fontsize=7, color=INK2,
                        va="center")
        for _, r in t.iterrows():
            ax.plot([r.AMI_country, r.AMI_group], [ypos[r.label]] * 2, color=GRID, lw=1.5,
                    zorder=2)
        ax.set_title(f"Term {term}", loc="left")
        ax.set_xlim(0, 1)
        ax.set_xlabel("AMI with label (0 = chance)")
        ax.set_yticks(range(len(all_labels)))
        ax.set_yticklabels([_wrap(x) for x in all_labels], fontsize=7.5)
    _top_legend(fig, np.atleast_1d(axes)[0])
    _save(fig, path)


def splits(tables, path=None):
    """tables: {term: DataFrame from rq1.splits_follow_nationality}"""
    order = [g for g in ["GUE_NGL", "GREEN_EFA", "SD", "RENEW", "EPP", "ECR", "ID", "PFE", "ESN"]
             if any(g in set(t.group) for t in tables.values())]
    y_of = {g: i for i, g in enumerate(order[::-1])}
    fig, axes = plt.subplots(1, len(tables), figsize=(3.6 * len(tables), 2.8), sharey=True)
    for ax, (term, t) in zip(np.atleast_1d(axes), tables.items()):
        t = t[t.group != "ALL (pooled)"]
        y = t.group.map(y_of).to_numpy()
        ax.barh(y + 0.2, t.together_same_party, height=0.38, color="#257de0",
                label="same national party")
        ax.barh(y - 0.2, t.together_different_party, height=0.38, color="#f47230",
                label="different national party")
        for g in order:
            if g not in set(t.group):
                ax.text(0.02, y_of[g], "not in this term", fontsize=7, color=INK2, va="center")
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([GROUP_NAMES[g] for g in order[::-1]], fontsize=7.5)
        ax.set_xlim(0, 1)
        ax.set_title(f"Term {term}", loc="left")
        ax.set_xlabel("Share of pairs in the same community")
        ax.grid(axis="y", visible=False)
    _top_legend(fig, np.atleast_1d(axes)[0])
    _save(fig, path)


def _top_legend(fig, ax):
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(l), fontsize=7.5,
               bbox_to_anchor=(0.5, 1.0))


def _spread(ys, gap):
    """Move label y-positions apart so no two are closer than gap (keeps order)."""
    idx = np.argsort(ys)
    out = np.array(ys, float)
    for _ in range(50):
        moved = False
        for a, b in zip(idx[:-1], idx[1:]):
            if out[b] - out[a] < gap:
                mid = (out[a] + out[b]) / 2
                out[a], out[b] = mid - gap / 2, mid + gap / 2
                moved = True
        if not moved:
            break
    return out


def _repel(pos, coords, fx=0.13, fy=0.07):
    """Nudge overlapping text labels apart vertically (fractions of the data range)."""
    rx = np.ptp(coords[:, 0]); ry = np.ptp(coords[:, 1])
    p = pos.copy()
    for _ in range(100):
        moved = False
        for i in range(len(p)):
            for j in range(i + 1, len(p)):
                if abs(p[i, 0] - p[j, 0]) < fx * rx and abs(p[i, 1] - p[j, 1]) < fy * ry:
                    s_ = 1 if p[i, 1] >= p[j, 1] else -1
                    p[i, 1] += s_ * 0.02 * ry; p[j, 1] -= s_ * 0.02 * ry
                    moved = True
        if not moved:
            break
    lo, hi = coords[:, 0].min() + 0.07 * rx, coords[:, 0].max() - 0.07 * rx
    p[:, 0] = np.clip(p[:, 0], lo, hi)        # keep labels inside the axes
    return p


def _wrap(s, n=24):
    words, lines, cur = s.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > n and cur:
            lines.append(cur); cur = w
        else:
            cur = (cur + " " + w).strip()
    return "\n".join(lines + [cur])


def _save(fig, path):
    fig.tight_layout()
    if path is not None:
        fig.savefig(path)
    return fig


def epp_pivot(by_half, power, path=None):
    """Left: per half-year, how often the EPP sides with the right bloc when the blocs
    disagree, and how often the right then wins. Right: share of votes each group could
    flip on its own, term 9 vs term 10."""
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.4, 2.9), gridspec_kw=dict(width_ratios=[1.35, 1]))
    h = by_half.reset_index(drop=True)
    x = np.arange(len(h))
    ax.plot(x, h.epp_with_right, color="#257de0", lw=2, marker="o", ms=3.5,
            label="EPP sides with the right bloc")
    ax.plot(x, h.right_wins_when_epp_joins, color="#f47230", lw=2, marker="o", ms=3.5,
            label="...and the right then wins")
    split = int(np.argmax(h.term.to_numpy() == 10))
    ax.axvline(split - 0.5, color=AXIS, lw=0.8, ls="--")
    ax.text(split - 0.4, 0.97, "2024 election", fontsize=7, color=INK2, va="top")
    ax.set_xticks(x[::2]); ax.set_xticklabels(h.half.iloc[::2], rotation=45, ha="right", fontsize=7)
    ax.set_ylim(0, 1); ax.set_ylabel("Share of votes where left\nand right blocs disagree")
    ax.set_title("EPP choices and their effect", loc="left")
    ax.legend(loc="lower left", fontsize=7)

    p = (power[power.group != "ESN"].set_index("group")
         [["pivotal_share_term9", "pivotal_share_term10"]].sort_values("pivotal_share_term10"))
    y = np.arange(len(p))
    ax2.barh(y + 0.2, p.pivotal_share_term9, height=0.38, color="#9dc8f7", label="Term 9")
    ax2.barh(y - 0.2, p.pivotal_share_term10, height=0.38, color="#1b64be", label="Term 10")
    names = {**GROUP_NAMES, "FAR": "ID / PfE"}
    ax2.set_yticks(y); ax2.set_yticklabels([names.get(g, g) for g in p.index], fontsize=7.5)
    ax2.set_xlabel("Share of votes the group could flip alone")
    ax2.set_title("Voting power", loc="left")
    ax2.grid(axis="y", visible=False)
    ax2.legend(loc="lower right", fontsize=7)
    _save(fig, path)


def dimensionality(by_half, path=None):
    h = by_half.dropna(subset=["dim1_share"]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    x = np.arange(len(h))
    ax.errorbar(x, h.dim1_share, yerr=2 * h.dim1_sd, color="#257de0", lw=2, marker="o",
                ms=3.5, capsize=0)
    split = int(np.argmax(h.term.to_numpy() == 10))
    ax.axvline(split - 0.5, color=AXIS, lw=0.8, ls="--")
    ax.text(split - 0.4, 0.49, "2024 election", fontsize=7, color=INK2, va="top")
    ax.set_xticks(x); ax.set_xticklabels(h.half, rotation=45, ha="right", fontsize=7)
    ax.set_ylim(0.25, 0.5)
    ax.set_ylabel("Variance on dimension 1")
    ax.set_title("How one-dimensional is voting?", loc="left")
    _save(fig, path)
