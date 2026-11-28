"""RQ1 on five terms, 2004-2026, from the merged dataset (build_merged.py): do voting
communities follow European groups, countries or national parties, and did that change?

Same settings as the notebook (config.py), same code (rq1.py, svd_map.py). Terms 9 and 10
use config.TERMS (term 9 from 1 Feb 2020, after Brexit), so their numbers should equal the
notebook's; terms 6-8 use the legislature dates. Run from the repo root:
    python src/all_terms_analysis.py          (~3 min)
Writes results/tables/all_terms_*.csv and results/figures/all_terms_rq1.png,
results/figures/all_terms_svd_maps.png.

Unlike the three-term comparison (ep8_analysis.py), national-party results are reported
for every term: the merged data has each MEP's party on the day of each vote.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import config
import rq1
import svd_map
import figures
from agreement import build_term_graph
from evaluation import describe_communities
from build_merged import load_merged

OUT = Path(__file__).resolve().parent.parent / "results"
TABLES, FIGS = OUT / "tables", OUT / "figures"
TERMS = [6, 7, 8, 9, 10]
TERM_NAMES = {6: "2004-09", 7: "2009-14", 8: "2014-19", 9: "2020-24", 10: "2024-26"}

# Groups that only exist before 2019, coloured like their closest successor
figures.GROUP_COLOURS.update({"ALDE": figures.GROUP_COLOURS["RENEW"],
                              "ENF": figures.GROUP_COLOURS["ID"], "EFDD": "#8c6d31",
                              "EFD": "#8c6d31", "IND_DEM": "#a0785a", "UEN": "#b07aa1",
                              "ITS": "#4b3b8f"})
figures.GROUP_NAMES.update({"ALDE": "ALDE", "ENF": "ENF", "EFDD": "EFDD", "EFD": "EFD",
                            "IND_DEM": "IND/DEM", "UEN": "UEN", "ITS": "ITS"})


def load():
    """Merged data with `term` set to the analysis periods: config.TERMS for 9 and 10,
    the legislatures for 6-8, 0 for votes outside them (Jul 2019 - Jan 2020)."""
    d = load_merged(columns=["vote_id", "member_id", "position", "country_code",
                             "group_code", "party"])
    v = d["votes"]
    term = v.term.where(v.term < 9, 0).astype("int8")
    for t, (lo, hi) in config.TERMS.items():
        term[(v.timestamp >= lo) & (v.timestamp < hi)] = t
    d["votes"] = v.assign(term=term)
    return d


def run_term(d, term):
    g = build_term_graph(d, term)
    comm, runs, _ = rq1.best_of_seeds(g["W"])
    main = rq1.main_table(g, comm, runs)
    main = main.reset_index().assign(term=term, Q_louvain=main.attrs["Q_louvain"],
                                     n_communities=main.attrs["n_communities"],
                                     n_meps=g["X"].shape[0], n_votes=g["X"].shape[1])
    coords, ex = svd_map.ideal_points(g["X"])
    coords = svd_map.orient(coords, g["labels"].group.to_numpy())
    out = dict(
        main=main,
        pairs=rq1.pair_agreement(g["agree"], g["labels"]).assign(term=term),
        communities=describe_communities(g["labels"], comm).assign(term=term),
        splits=rq1.splits_follow_nationality(g["labels"], comm).assign(term=term),
        sweep=rq1.resolution_sweep(g, config.GAMMA_SWEEP).assign(term=term),
        robust=rq1.robustness(g).assign(term=term),
        sep=svd_map.separation(coords, g["labels"]).assign(
            term=term, explained_dim1=ex[0], explained_dim2=ex[1]),
        arch=pd.DataFrame([dict(term=term, **svd_map.arch_check(coords))]),
    )
    a = main.set_index("label").AMI
    print(f"term {term} ({TERM_NAMES[term]}): {g['X'].shape[0]} MEPs x {g['X'].shape[1]} "
          f"contested votes, {main.n_communities.iloc[0]} communities, AMI group "
          f"{a['group']:.3f} / party {a['party']:.3f} / country {a['country']:.3f}")
    return out, (coords, g["labels"], ex)


def reproduction_check(main):
    """Terms 9-10 must equal the notebook's saved RQ1 results (same data, same code);
    term 8 should be close to the separate VoteWatch EP8 pipeline (ep8_analysis.py)."""
    rows = []
    for name, terms in [("rq1_main", (9, 10)), ("ep8_rq1_main", (8,))]:
        f = TABLES / f"{name}.csv"
        if not f.exists():
            continue
        old = pd.read_csv(f)
        for t in terms:
            for lab in ["group", "country", "party"]:
                o = old[(old.term == t) & (old.label == lab)]
                n = main[(main.term == t) & (main.label == lab)]
                if len(o) and len(n):
                    rows.append(dict(term=t, label=lab, saved_in=name, AMI_saved=o.AMI.iloc[0],
                                     AMI_now=n.AMI.iloc[0],
                                     diff=n.AMI.iloc[0] - o.AMI.iloc[0]))
    r = pd.DataFrame(rows)
    print("\nreproduction of saved results (AMI):")
    print(r.round(4).to_string(index=False))
    return r


def summary(main, pairs, sep, arch, splits):
    rows = {}
    for t in TERMS:
        m, p = main[main.term == t].set_index("label"), pairs[pairs.term == t].set_index("pair_type")
        s, a = sep[sep.term == t].set_index("label"), arch[arch.term == t].iloc[0]
        sp = splits[(splits.term == t) & (splits.group == "ALL (pooled)")].iloc[0]
        rows[TERM_NAMES[t]] = {
            "MEPs / contested votes": f"{m.n_meps.iloc[0]} / {m.n_votes.iloc[0]}",
            "communities found (Louvain)": int(m.n_communities.iloc[0]),
            "modularity": m.Q_louvain.iloc[0],
            "AMI with European group": m.AMI["group"],
            "AMI with national party": m.AMI["party"],
            "AMI with country": m.AMI["country"],
            "agreement: same group, same party": p.mean_agreement["same group, same national party"],
            "agreement: same group, different country": p.mean_agreement["same group, different country"],
            "agreement: different group, same country": p.mean_agreement["different group, same country"],
            "agreement: baseline (different group and country)":
                p.mean_agreement["different group, different country (baseline)"],
            "within-group splits: together, same party": sp.together_same_party,
            "within-group splits: together, different party": sp.together_different_party,
            "SVD map: variance on dimension 1": s.explained_dim1.iloc[0],
            "SVD map: variance on dimension 2": s.explained_dim2.iloc[0],
            "SVD map: dim 2 explained by a curve of dim 1 (R2)": a.R2_quadratic,
            "SVD map: silhouette by group": s.silhouette["group"],
            "SVD map: silhouette by country": s.silhouette["country"],
        }
    return pd.DataFrame(rows)


PAIR_ORDER = [("same group, same national party", "Same group,\nsame party"),
              ("same group, different country", "Same group,\ndifferent country"),
              ("different group, same country", "Different group,\nsame country"),
              ("different group, different country (baseline)", "Different group,\ndifferent country")]


def figure(main, pairs, path=None):
    """Left: AMI of the communities with group / party / country per term. Right: mean
    agreement of MEP pairs by what they share, per term."""
    import matplotlib.pyplot as plt
    x = np.arange(len(TERMS))
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw={"width_ratios": [1, 1.6]})
    for col in ["group", "party", "country"]:
        y = [main[(main.term == t) & (main.label == col)].AMI.iloc[0] for t in TERMS]
        a.plot(x, y, color=figures.LABEL_COLOURS[col], lw=2, marker="o", ms=5)
        a.annotate(figures.LABEL_NAMES[col], (x[-1], y[-1]), xytext=(6, 0),
                   textcoords="offset points", va="center", fontsize=8, color=figures.INK2,
                   annotation_clip=False)
    a.axvline(2.5, color=figures.AXIS, lw=0.8, ls="--")
    a.text(2.5, 0.97, " HowTheyVote →", fontsize=7, color=figures.INK2, va="top")
    a.text(2.5, 0.97, "VoteWatch ", fontsize=7, color=figures.INK2, va="top", ha="right")
    a.set_xticks(x, [TERM_NAMES[t] for t in TERMS], fontsize=8)
    a.set_ylim(0, 1); a.set_xlim(-0.3, len(TERMS) - 0.3)
    a.set_ylabel("AMI with label (0 = chance)")
    a.set_title("Which label do the voting communities follow?", loc="left")
    width = 0.8 / len(TERMS)
    shades = ["#dbe8f8", "#b9d3f2", "#8bb8ec", "#5797e4", "#257de0"]
    for i, t in enumerate(TERMS):
        p = pairs[pairs.term == t].set_index("pair_type").mean_agreement
        vals = [p.get(k, np.nan) for k, _ in PAIR_ORDER]
        b.bar(np.arange(len(PAIR_ORDER)) + (i - (len(TERMS) - 1) / 2) * width, vals,
              width=width * 0.95, color=shades[i], label=TERM_NAMES[t])
    b.set_xticks(np.arange(len(PAIR_ORDER)), [lbl for _, lbl in PAIR_ORDER], fontsize=7.5)
    b.set_ylim(0.4, 1.09); b.set_yticks(np.arange(0.4, 1.01, 0.1))
    b.set_ylabel("Mean agreement of MEP pairs")
    b.set_title("How often two MEPs vote the same way", loc="left")
    b.legend(ncol=len(TERMS), fontsize=7, loc="upper right")
    return figures._save(fig, path)


def main():
    pd.set_option("display.width", 220)
    d = load()
    res, maps = {}, {}
    for t in TERMS:
        res[t], maps[TERM_NAMES[t]] = run_term(d, t)
    tables = {k: pd.concat([res[t][k] for t in TERMS], ignore_index=True) for k in res[TERMS[0]]}
    for k, df in tables.items():
        df.to_csv(TABLES / f"all_terms_{k}.csv", index=False)
    repro = reproduction_check(tables["main"])
    repro.to_csv(TABLES / "all_terms_reproduction.csv", index=False)
    s = summary(tables["main"], tables["pairs"], tables["sep"], tables["arch"], tables["splits"])
    s.to_csv(TABLES / "all_terms_summary.csv")
    print("\n", s.round(3).to_string())
    figure(tables["main"], tables["pairs"], FIGS / "all_terms_rq1.png")
    figures.svd_maps(maps, FIGS / "all_terms_svd_maps.png")
    print("\nsaved results/tables/all_terms_*.csv, results/figures/all_terms_rq1.png, "
          "all_terms_svd_maps.png")


if __name__ == "__main__":
    main()
