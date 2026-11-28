"""RQ2 on five terms, 2004-2026, from the merged dataset: which coalitions pass votes,
who rebels together, and the EPP's role between the left and right blocs.

Same code as the notebook (rq2.py, frequent_items.py) with terms=(6, ..., 10). Terms 9 and
10 use config.TERMS, so their rows should equal the notebook's saved rq2_* tables (checked
at the end). The right bloc is every group to the EPP's right in each term (rq2.RIGHT_BLOC);
the left bloc is S&D + Greens/EFA + The Left throughout. EP6 is left out of the
"legislation (COD)" subset: its file gives the procedure type for only ~21% of votes.
Run from the repo root:   python src/all_terms_rq2.py      (~2 min)
Writes results/tables/all_terms_rq2_*.csv and results/figures/all_terms_rq2.png.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import rq2
import figures
import frequent_items as fi
from all_terms_analysis import load, TERMS, TERM_NAMES

OUT = Path(__file__).resolve().parent.parent / "results"
TABLES, FIGS = OUT / "tables", OUT / "figures"


def save(df, name):
    df.to_csv(TABLES / f"all_terms_rq2_{name}.csv", index=False)
    return df


def baskets_and_algorithms(d):
    """Basket sizes per term, and A-priori vs PCY on every term's baskets (the two must
    find the same itemsets; compare_algorithms asserts it)."""
    B, party_group, stats = {}, {}, []
    for t in TERMS:
        B[f"FOR, term {t}"] = fi.build_baskets(d, t, level="party", passed_only=True)[0]
        B[f"rebels, term {t}"], _, party_group[t] = rq2.rebel_baskets(d, t)
    for k, v in B.items():
        stats.append(dict(basket_set=k, **fi.basket_stats(v)))
    cmp = pd.concat([
        rq2.compare_algorithms({k: v for k, v in B.items() if k.startswith("FOR")}, [0.5]),
        rq2.compare_algorithms({k: v for k, v in B.items() if k.startswith("rebels")}, [0.01])])
    return B, party_group, pd.DataFrame(stats), cmp


def rebels(B, party_group):
    out = []
    for t in TERMS:
        summ, _ = rq2.rebel_pair_summary(B[f"rebels, term {t}"], party_group[t])
        out.append(summ.assign(term=t))
    return pd.concat(out, ignore_index=True)


def epp_pivot(d):
    by_term, by_half = rq2.epp_pivot(d)
    cod = by_term.subset == "legislation (COD)"
    by_term = by_term[~(cod & (by_term.term == 6))].reset_index(drop=True)
    return by_term, by_half


def reproduction_check(win_summary, coalitions, pivot, power):
    """Terms 9-10 rows vs the notebook's saved rq2 tables."""
    rows = []

    def cmp(name, new, old, cols):
        a, b = new.reset_index(drop=True), old.reset_index(drop=True)
        same = len(a) == len(b) and all(np.allclose(a[c].to_numpy(float), b[c].to_numpy(float),
                                                    equal_nan=True) for c in cols)
        rows.append(dict(table=name, rows=len(b), identical=same))

    f = lambda n: pd.read_csv(TABLES / f"{n}.csv")
    cmp("winning coalitions summary", win_summary[win_summary.term.isin([9, 10])],
        f("rq2_winning_coalitions_summary"), ["coalitions", "cross_border", "cross_group"])
    old = f("rq2_epp_pivot")
    new = pivot[pivot.term.isin([9, 10])].merge(old[["subset", "term"]], on=["subset", "term"])
    cmp("EPP pivot", new.sort_values(["subset", "term"]), old.sort_values(["subset", "term"]),
        ["votes", "epp_with_right", "right_wins_when_epp_joins"])
    old = f("rq2_voting_power").set_index("group")
    new = power.set_index("group").reindex(old.index)
    cmp("voting power", new, old, ["pivotal_share_term9", "pivotal_share_term10"])
    old = f("rq2_group_coalitions")
    new = coalitions.set_index("itemset").reindex(old.itemset).reset_index()
    cmp("group coalitions (support in terms 9, 10)", new, old,
        ["support_term9", "support_term10"])
    r = pd.DataFrame(rows)
    print("\nreproduction of the notebook's saved RQ2 tables (terms 9-10):")
    print(r.to_string(index=False))
    return r


def figure(win_summary, pivot, power, path=None):
    """Left: share of winning party pairs that cross group lines. Middle: when left and
    right disagree, how often the EPP sides with the right, and how often the right wins
    with / without it. Right: share of contested votes each group could flip alone."""
    import matplotlib.pyplot as plt
    x = np.arange(len(TERMS))
    names = [TERM_NAMES[t] for t in TERMS]
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(13, 3.6))

    w = win_summary[win_summary["size"] == 2].set_index("term").reindex(TERMS)
    a.plot(x, w.cross_group, color=figures.GROUP_COLOURS["EPP"], lw=2, marker="o", ms=5)
    a.set_ylim(0, 1)
    a.set_title("Winning party pairs that cross\ngroup lines", loc="left")
    a.set_ylabel("Share of frequent winning pairs")

    p = pivot[pivot.subset == "all contested votes"].set_index("term").reindex(TERMS)
    for col, lab, colour in [("epp_with_right", "EPP sides\nwith right", figures.INK),
                             ("right_wins_when_epp_joins", "Right wins,\nEPP with it",
                              figures.GROUP_COLOURS["ECR"]),
                             ("right_wins_when_epp_against", "Right wins,\nEPP against",
                              figures.GROUP_COLOURS["SD"])]:
        b.plot(x, p[col], lw=2, marker="o", ms=5, color=colour)
        b.annotate(lab, (x[-1], p[col].iloc[-1]), xytext=(6, 0), textcoords="offset points",
                   va="center", fontsize=7, color=figures.INK2, annotation_clip=False)
    b.set_ylim(-0.02, 1)
    b.set_title("Left and right blocs disagree:\nthe EPP decides", loc="left")
    b.set_ylabel("Share of those votes")

    pw = power.set_index("group")
    lines = {"EPP": ["EPP"], "S&D": ["SD"], "ALDE / Renew": ["ALDE", "RENEW"],
             "Greens/EFA": ["GREEN_EFA"], "ECR": ["ECR"]}
    colours = {"EPP": "EPP", "S&D": "SD", "ALDE / Renew": "RENEW", "Greens/EFA": "GREEN_EFA",
               "ECR": "ECR"}
    ends = {}
    for lab, gs in lines.items():
        y = []
        for t in TERMS:
            vals = [pw.loc[g, f"pivotal_share_term{t}"] for g in gs
                    if g in pw.index and pd.notna(pw.loc[g, f"pivotal_share_term{t}"])]
            y.append(vals[0] if vals else np.nan)
        y = np.array(y)
        ok = ~np.isnan(y)
        c.plot(x[ok], y[ok], lw=2, marker="o", ms=5, color=figures.GROUP_COLOURS[colours[lab]])
        ends[lab] = y[ok][-1]
    for (lab, y_end), y_lab in zip(ends.items(), figures._spread(list(ends.values()), 0.035)):
        c.annotate(lab, (x[-1], y_end), xytext=(x[-1] + 0.15, y_lab), textcoords="data",
                   va="center", fontsize=7, color=figures.INK2, annotation_clip=False)
    c.set_ylim(0, None)
    c.set_title("Votes a group could flip by\nswitching sides alone", loc="left")
    c.set_ylabel("Share of contested votes")
    for ax in (a, b, c):
        ax.set_xticks(x, names, fontsize=7.5, rotation=0)
        ax.set_xlim(-0.3, len(TERMS) - 0.3)
        ax.axvline(2.5, color=figures.AXIS, lw=0.8, ls="--")
    return figures._save(fig, path)


def main():
    pd.set_option("display.width", 220)
    d = load()
    B, party_group, stats, cmp = baskets_and_algorithms(d)
    save(stats, "basket_stats")
    save(cmp, "apriori_vs_pcy")
    print("A-priori and PCY found the same itemsets on all", len(cmp), "basket sets")

    items, win_summary, mix = rq2.winning_party_coalitions(d, support=0.5, max_size=3,
                                                           terms=TERMS)
    border, group = rq2.top_coalitions(items, size=2, top=10)
    save(win_summary, "winning_coalitions_summary")
    save(mix, "winning_coalitions_group_mix")
    save(pd.concat([border.assign(kind="cross-border"), group.assign(kind="cross-group")]),
         "winning_coalitions_top_pairs")
    rebel = save(rebels(B, party_group), "rebel_pair_summary")
    coalitions = save(rq2.group_coalitions(d, terms=TERMS), "group_coalitions")
    pivot, pivot_half = epp_pivot(d)
    save(pivot, "epp_pivot"); save(pivot_half, "epp_pivot_by_half")
    power = save(rq2.voting_power(d, terms=TERMS), "voting_power")
    save(reproduction_check(win_summary, coalitions, pivot, power), "reproduction")

    print("\nwinning party coalitions (support 0.5):")
    print(win_summary.round(3).to_string(index=False))
    print("\nlargest winning coalitions (3 parties) by group mix, top 8:")
    print(mix.head(8).round(3).to_string(index=False))
    print("\nwho rebels together:")
    print(rebel.pivot_table(index="pair_type", columns="term",
                            values=["share_of_co_rebellions", "median_lift"]).round(3).to_string())
    print("\ngroup coalitions on adopted votes (support), top 12 by 2024-26:")
    cols = ["itemset"] + [f"support_term{t}" for t in TERMS]
    print(coalitions[cols].head(12).round(3).to_string(index=False))
    print("\nEPP between the blocs:")
    print(pivot.round(3).to_string(index=False))
    print("\nvoting power (share of contested votes a group could flip alone):")
    print(power[["group"] + [f"pivotal_share_term{t}" for t in TERMS]].round(3).to_string(index=False))
    figure(win_summary, pivot, power, FIGS / "all_terms_rq2.png")
    print("\nsaved results/tables/all_terms_rq2_*.csv and results/figures/all_terms_rq2.png")


if __name__ == "__main__":
    main()
