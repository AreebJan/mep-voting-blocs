"""The rest of the notebook's pipeline on five terms, 2004-2026, from the merged dataset:
data overview, Louvain vs networkx, RQ1 by policy area, SVD group centres, "do the
extremes vote together?", dimensionality per half-year, the PCY bucket sweep, and the
notebook's resolution-sweep / splits / policy-area figures for all terms.
(all_terms_analysis.py, all_terms_rq2.py and all_terms_rq3.py cover RQ1-RQ3.)

Two steps differ before 2019:
- policy areas: VoteWatch's own policy areas for terms 6-8 (21 areas), HowTheyVote's
  top-level OEIL subjects for terms 9-10. Compare areas within a term, not across sources.
- extremes: centre = EPP + S&D + the liberals (ALDE before 2019, Renew after); the far
  right = the RQ2 right bloc without ECR: UEN + IND/DEM + ITS (2004-09), EFD (2009-14),
  EFDD + ENF (2014-19). Terms 9-10 keep the notebook's ID and PfE.
Terms 9-10 are checked against the notebook's saved tables at the end. Not repeated: the
HowTheyVote "groups required for majority" check (that data only exists for term 10).
Run from the repo root:   python src/all_terms_rest.py      (~5 min)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import config
import rq1
import rq2
import svd_map
import figures
from paths import PROC
from agreement import build_term_graph
from all_terms_analysis import load, TERMS, TERM_NAMES

OUT = Path(__file__).resolve().parent.parent / "results"
TABLES, FIGS = OUT / "tables", OUT / "figures"

EXTREMES = {   # term: (centre groups, far-right groups)
    6: (["EPP", "SD", "ALDE"], ["UEN", "IND_DEM", "ITS"]),
    7: (["EPP", "SD", "ALDE"], ["EFD"]),
    8: (["EPP", "SD", "ALDE"], ["EFDD", "ENF"]),
    9: (["EPP", "SD", "RENEW"], ["ID"]),
    10: (["EPP", "SD", "RENEW"], ["PFE"]),
}


def save(df, name):
    df.to_csv(TABLES / f"all_terms_{name}.csv", index=False)
    return df


def add_topics(d):
    """vote_topics for rq1.policy_areas: OEIL subjects (HowTheyVote) and VoteWatch's
    policy areas, with `topic` as text so both fit in one column."""
    htv = pd.read_parquet(PROC / "vote_topics.parquet")
    htv = htv.assign(topic=htv.topic.astype(str), source="OEIL subject")
    v = d["votes"]
    vw = v[v.vote_id >= config.VW_VOTE_ID_OFFSET].dropna(subset=["policy_area"])
    vw = pd.DataFrame({"vote_id": vw.vote_id, "topic": "VW:" + vw.policy_area,
                       "label": vw.policy_area, "source": "VoteWatch policy area"})
    d["vote_topics"] = pd.concat([htv[["vote_id", "topic", "label", "source"]], vw],
                                 ignore_index=True)
    return d


def topics_for(d, term):
    """d with only the policy categories of the term's source (VoteWatch areas for
    terms 6-8, OEIL subjects for 9-10)."""
    src = "VoteWatch policy area" if term <= 8 else "OEIL subject"
    vt = d["vote_topics"]
    return dict(d, vote_topics=vt[vt.source == src])


def overview(d):
    v, mv = d["votes"], d["member_votes"]
    t = v[v.term > 0]
    o = t.groupby("term").agg(votes=("vote_id", "size"), contested=("contested", "sum"),
                              first=("timestamp", "min"), last=("timestamp", "max"),
                              source=("source", lambda s: "/".join(sorted(set(map(str, s))))))
    x = mv[mv.vote_id.isin(t.vote_id)]
    x = x.assign(term=x.vote_id.map(t.set_index("vote_id").term))
    o["meps"] = x.groupby("term").member_id.nunique()
    o["mep_votes"] = x.groupby("term").size()
    o["share_for_against"] = x.position.isin(config.POSITIONS).groupby(x.term).mean()
    return o.reset_index()


def extremes(d, term):
    """svd_map.extremes_co_opposition with the centre and far right of each term. On
    votes where the centre groups agree, how often do The Left and the far right both
    vote against them, compared to if they were independent?"""
    centre, far = EXTREMES[term]
    V, mv = d["votes"], d["member_votes"]
    v = V[(V.term == term) & V.contested]
    x = mv[mv.vote_id.isin(v.vote_id) & mv.position.isin(config.POSITIONS)]
    x = pd.DataFrame({"vote_id": x.vote_id.to_numpy(), "group": x.group_code.astype(str).to_numpy(),
                      "pos": np.where(x.position.to_numpy() == "FOR", 1, -1)})
    net = x.groupby(["vote_id", "group"]).pos.sum().unstack(fill_value=0)
    m = np.sign(net)
    farm = np.sign(net[[g for g in far if g in net]].sum(axis=1))
    unity = (m[centre].nunique(axis=1) == 1) & (m[centre[0]] != 0)
    c, f = m[unity], farm[unity]
    left_opp = (c.GUE_NGL == -c[centre[0]]).mean()
    right_opp = (f == -c[centre[0]]).mean()
    both = ((c.GUE_NGL == -c[centre[0]]) & (f == -c[centre[0]])).mean()
    agree = ((m.GUE_NGL == farm) & (m.GUE_NGL != 0)).sum() / ((m.GUE_NGL != 0) & (farm != 0)).sum()
    return dict(term=term, centre="+".join(centre), far_right="+".join(far),
                centre_united_share=unity.mean(), left_opposes=left_opp,
                far_right_opposes=right_opp, both_oppose=both,
                both_if_independent=left_opp * right_opp,
                ratio_to_independent=both / (left_opp * right_opp),
                left_far_right_majority_agreement=agree)


def dimensionality_figure(dims, path=None):
    import matplotlib.pyplot as plt
    h = dims.dropna(subset=["dim1_share"]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, 3))
    x = np.arange(len(h))
    ax.errorbar(x, h.dim1_share, yerr=2 * h.dim1_sd, color=figures.GROUP_COLOURS["EPP"], lw=2,
                marker="o", ms=3.5, capsize=0)
    for t in TERMS[1:]:
        first = np.argmax(h.term.to_numpy() == t)
        ax.axvline(first - 0.5, color=figures.AXIS, lw=0.8, ls="--")
    for t in TERMS:
        idx = np.where(h.term.to_numpy() == t)[0]
        if len(idx):
            ax.text(idx.mean(), 0.53, TERM_NAMES[t], ha="center", fontsize=7.5,
                    color=figures.INK2)
    ax.set_xticks(x[::2]); ax.set_xticklabels(h.half[::2], rotation=45, ha="right", fontsize=7)
    ax.set_ylim(0.15, 0.55)
    ax.set_ylabel("Variance on dimension 1")
    ax.set_title("How one-dimensional is voting? (500 random contested votes per half-year)",
                 loc="left")
    return figures._save(fig, path)


def reproduction_check(val, policy, centres, ext, buckets):
    """Terms 9-10 against the notebook's saved tables."""
    f = lambda n: pd.read_csv(TABLES / f"{n}.csv")
    rows = []

    def cmp(name, new, old, key, cols):
        m = old.merge(new, on=key, suffixes=("_old", "_new"))
        ok = len(m) == len(old) and all(np.allclose(m[f"{c}_old"].astype(float),
                                                    m[f"{c}_new"].astype(float), equal_nan=True)
                                        for c in cols)
        rows.append(dict(table=name, rows=len(old), matched=len(m), identical=ok))

    cmp("Louvain validation", val, f("louvain_validation"), ["term"],
        ["Q_ours", "Q_networkx", "NMI_ours_vs_networkx"])
    old = f("rq1_policy_areas")
    cmp("policy areas", policy[policy.term.isin([9, 10])].assign(label=lambda x: x.label),
        old, ["term", "label"], ["n_votes", "AMI_group", "AMI_country"])
    cmp("SVD group centres", centres, f("svd_group_centres"), ["term", "group"], ["dim1", "dim2"])
    cmp("extremes", ext, f("svd_extremes_co_opposition"), ["term"],
        ["left_opposes", "far_right_opposes", "ratio_to_independent"])
    old = f("rq2_pcy_bucket_sweep")
    old = old[old.baskets.str.startswith("rebels")].assign(term=9)
    cmp("PCY bucket sweep (rebels, term 9)", buckets, old, ["term", "n_buckets"],
        ["pcy_pair_candidates", "frequent_bucket_share"])
    r = pd.DataFrame(rows)
    print("\nreproduction of the notebook's saved tables (terms 9-10):")
    print(r.to_string(index=False))
    return r


def main():
    pd.set_option("display.width", 220)
    d = add_topics(load())
    ov = save(overview(d), "data_overview")
    print(ov.to_string(index=False))

    val, policy, centres, buckets = [], [], [], []
    for t in TERMS:
        g = build_term_graph(d, t)
        _, summary, _ = rq1.validate_louvain(g["W"])
        val.append(dict(term=t, **summary))
        policy.append(rq1.policy_areas(topics_for(d, t), t).assign(term=t))
        c, _ = svd_map.ideal_points(g["X"])
        c = svd_map.orient(c, g["labels"].group.to_numpy())
        centres.append(svd_map.group_centres(c, g["labels"]).reset_index().assign(term=t))
        rebels, _, _ = rq2.rebel_baskets(d, t)
        buckets.append(rq2.bucket_sweep(rebels, 0.01).assign(term=t))
        print(f"term {t}: done")
    val = save(pd.DataFrame(val), "louvain_validation")
    policy = save(pd.concat(policy, ignore_index=True), "policy_areas")
    centres = save(pd.concat(centres, ignore_index=True), "svd_group_centres")
    buckets = save(pd.concat(buckets, ignore_index=True), "pcy_bucket_sweep")
    ext = save(pd.DataFrame([extremes(d, t) for t in TERMS]), "svd_extremes_co_opposition")
    dims = save(svd_map.dimensionality_over_time(d), "svd_dimensionality_over_time")
    save(reproduction_check(val, policy, centres, ext, buckets[buckets.term == 9]),
         "rest_reproduction")

    print("\nLouvain vs networkx:")
    print(val[["term", "Q_ours", "Q_networkx", "NMI_ours_vs_networkx", "NMI_ours_vs_ours",
               "time_ours_s", "time_networkx_s"]].round(3).to_string(index=False))
    print("\nby policy area (AMI with group / country), terms 6-8 on VoteWatch areas:")
    p = policy[~policy.skipped.astype(bool)]
    print(p.groupby("term")[["AMI_group", "AMI_party", "AMI_country"]].agg(["min", "median", "max"])
          .round(3).to_string())
    print("areas where country beats group:",
          p[p.AMI_country > p.AMI_group][["term", "label"]].values.tolist())
    print("\nextremes:")
    print(ext.round(3).to_string(index=False))
    print("\ndimension 1 share by term (mean over half-years):",
          dims.groupby("term").dim1_share.mean().round(3).to_dict())

    # figures: the notebook's figures with all five terms
    sweep = pd.read_csv(TABLES / "all_terms_sweep.csv")
    figures.resolution_sweep({TERM_NAMES[t]: s for t, s in sweep.groupby("term")},
                             FIGS / "all_terms_resolution_sweep.png")
    splits = pd.read_csv(TABLES / "all_terms_splits.csv")
    figures.splits({TERM_NAMES[t]: s for t, s in splits.groupby("term")},
                   FIGS / "all_terms_splits_nationality.png")
    figures.policy_areas({TERM_NAMES[t]: s for t, s in policy[policy.term <= 8].groupby("term")},
                         FIGS / "all_terms_policy_areas_2004_2019.png")
    dimensionality_figure(dims, FIGS / "all_terms_dimensionality.png")
    print("\nsaved results/tables/all_terms_*.csv and results/figures/all_terms_*.png")


if __name__ == "__main__":
    main()
