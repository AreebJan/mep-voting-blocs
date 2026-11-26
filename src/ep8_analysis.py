"""RQ1 and the SVD map for the 2014-2019 term (EP8, VoteWatch data), with the same
settings as terms 9 and 10, and a comparison with their results.

Run from the repo root after python src/load_votewatch.py:
    python src/ep8_analysis.py
Writes results/tables/ep8_*.csv and results/figures/ep8_svd_map.png.
Terms 9 and 10 are read from the tables analysis.ipynb already saved.
"""
from pathlib import Path
import pandas as pd
import config
import rq1
import svd_map
import figures
from agreement import build_term_graph
from evaluation import describe_communities
from load_votewatch import load_ep8, TERM

OUT = Path(__file__).resolve().parent.parent / "results"
TABLES, FIGS = OUT / "tables", OUT / "figures"

# EP8 groups that no longer exist: ALDE became Renew, ENF became ID
figures.GROUP_COLOURS.update({"ALDE": figures.GROUP_COLOURS["RENEW"],
                              "ENF": figures.GROUP_COLOURS["ID"], "EFDD": "#8c6d31"})
figures.GROUP_NAMES.update({"ALDE": "ALDE", "ENF": "ENF", "EFDD": "EFDD"})


def save(df, name):
    df.to_csv(TABLES / f"ep8_{name}.csv", index=False)
    return df


def run():
    d = load_ep8()
    g = build_term_graph(d, TERM)
    print(f"term {TERM}: {g['X'].shape[0]} MEPs x {g['X'].shape[1]} contested votes, "
          f"{int((g['W'] > 0).sum() // 2):,} edges")

    comm, runs, _ = rq1.best_of_seeds(g["W"])
    main = rq1.main_table(g, comm, runs)
    main = save(main.reset_index().assign(term=TERM, Q_louvain=main.attrs["Q_louvain"],
                                          n_communities=main.attrs["n_communities"]), "rq1_main")
    pairs = save(rq1.pair_agreement(g["agree"], g["labels"]).assign(term=TERM), "rq1_pair_agreement")
    save(describe_communities(g["labels"], comm).assign(term=TERM), "rq1_communities")
    splits = save(rq1.splits_follow_nationality(g["labels"], comm).assign(term=TERM),
                  "rq1_splits_nationality")
    sweep = save(rq1.resolution_sweep(g, config.GAMMA_SWEEP).assign(term=TERM),
                 "rq1_resolution_sweep")
    robust = save(rq1.robustness(g).assign(term=TERM), "rq1_robustness")

    coords, ex = svd_map.ideal_points(g["X"])
    coords = svd_map.orient(coords, g["labels"].group.to_numpy())
    sep = save(svd_map.separation(coords, g["labels"]).assign(
        term=TERM, explained_dim1=ex[0], explained_dim2=ex[1]), "svd_separation")
    save(pd.DataFrame([dict(term=TERM, **svd_map.arch_check(coords))]), "svd_arch_check")
    figures.svd_maps({TERM: (coords, g["labels"], ex)}, FIGS / "ep8_svd_map.png")
    return dict(main=main, pairs=pairs, splits=splits, sweep=sweep, robust=robust, sep=sep)


def compare_terms(r):
    """EP8 next to the saved results for terms 9 and 10."""
    def both(new, old_name):
        old = pd.read_csv(TABLES / f"{old_name}.csv")
        return pd.concat([new, old], ignore_index=True)

    main = both(r["main"], "rq1_main")
    main = main.pivot_table(index="term", columns="label", values=["AMI", "ARI"]).round(3)
    pairs = both(r["pairs"], "rq1_pair_agreement").pivot_table(
        index="pair_type", columns="term", values="mean_agreement").round(3)
    sep = both(r["sep"], "svd_separation").pivot_table(
        index="term", columns="label", values="silhouette").round(3)
    splits = both(r["splits"], "rq1_splits_nationality")
    splits = splits[splits.group == "ALL (pooled)"].set_index("term")[
        ["together_same_party", "together_different_party"]].round(3)
    for name, t in [("main", main), ("pair_agreement", pairs), ("svd_separation", sep),
                    ("splits", splits)]:
        t.to_csv(TABLES / f"three_terms_{name}.csv")
    return main, pairs, sep, splits


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    results = run()
    for t in compare_terms(results):
        print(t, "\n")
