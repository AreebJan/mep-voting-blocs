"""RQ3 on five terms, 2004-2026, from the merged dataset: using only votes cast before a
switch, did MEPs who later changed group already vote differently from those who stayed?

Same test as the notebook (rq3_switchers.py). Group memberships with dates: HowTheyVote's
for terms 9-10 (so their events and AUCs equal the notebook's, checked at the end), the
EP's records (ep_group_history.csv) for terms 6-8. Each MEP's group on each vote is the
merged data's per-vote group.

Groups that dissolved (no members left the day after, e.g. ITS in Nov 2007 and EFDD's
4-day collapse in Oct 2014) are not individual switches: those events are left out of the
main result and kept in a sensitivity run with all events.
Run from the repo root:   python src/all_terms_rq3.py      (~2 min)
Writes results/tables/all_terms_rq3_*.csv and results/figures/all_terms_rq3.png.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import rq3_switchers as rq3
import figures
from paths import RAW
from config import EP_TERM_STARTS, EP_GROUP_CODES, GROUP_ALIASES, N_BOOTSTRAP
from build_merged import load_merged
from all_terms_analysis import TERMS, TERM_NAMES

OUT = Path(__file__).resolve().parent.parent / "results"
TABLES, FIGS = OUT / "tables", OUT / "figures"
NOTEBOOK_TERMS = dict(rq3.MEMBERSHIP_TERMS)          # terms 9-10, as in the notebook


def term_windows():
    starts = list(EP_TERM_STARTS.items())
    w = {t: (s, starts[i + 1][1]) for i, (t, s) in enumerate(starts[:-1])}
    w.update(NOTEBOOK_TERMS)
    return w


def memberships():
    """Dated group memberships in HowTheyVote's format: EP records for terms 6-8 (clipped
    to each term), HowTheyVote's own table for terms 9-10."""
    h = pd.read_csv(RAW / "ep_group_history.csv", parse_dates=["start_date", "end_date"])
    h = h[h.kind == "EU_POLITICAL_GROUP"]
    parts = []
    for t, (lo, hi) in term_windows().items():
        if t >= 9:
            continue
        g = h[(h.start_date < hi) & (h.end_date.isna() | (h.end_date >= lo))].copy()
        unknown = set(g.org_label) - set(EP_GROUP_CODES)
        assert not unknown, f"add to config.EP_GROUP_CODES: {unknown}"
        parts.append(g.assign(group_code=g.org_label.map(EP_GROUP_CODES), term=t))
    htv = pd.read_csv(RAW / "group_memberships.csv.gz", parse_dates=["start_date", "end_date"])
    htv["group_code"] = htv.group_code.replace(GROUP_ALIASES)
    cols = ["member_id", "group_code", "term", "start_date", "end_date"]
    return pd.concat([p[cols] for p in parts] + [htv[cols]], ignore_index=True)


def dissolutions(gm):
    """Event clusters whose origin group had no members left the day after the switch."""
    ev = rq3.switch_events(gm)
    out = []
    for cl, e in ev.groupby("cluster"):
        o, day = e.origin.iloc[0], e.date.iloc[0] + pd.Timedelta(days=1)
        if o in rq3.NOT_A_GROUP:
            continue
        left = gm[(gm.group_code == o) & (gm.start_date <= day) &
                  (gm.end_date.isna() | (gm.end_date >= day))].member_id.nunique()
        if left == 0:
            out.append(dict(cluster=cl, origin=o, date=e.date.iloc[0].date(), events=len(e)))
    return pd.DataFrame(out)


def by_term(per_event, seed=0):
    rows = []
    for col in ["leave_auc", "join_auc"]:
        for t in TERMS:
            x = per_event[per_event.term == t]
            if x[col].notna().sum() >= 3:
                rows.append(dict(term=t, **rq3._summarise(x, col, N_BOOTSTRAP, seed)))
            else:
                rows.append(dict(term=t, test=col.replace("_auc", ""),
                                 events=int(x[col].notna().sum())))
    return pd.DataFrame(rows).drop(columns="by_term", errors="ignore")


def reproduction_check(per_event):
    """Terms 9-10 vs the notebook's saved rq3 tables."""
    old_ev = pd.read_csv(TABLES / "rq3_events.csv")
    new_ev = per_event[per_event.term.isin([9, 10])]
    k = ["member_id", "term", "origin", "destination", "status"]
    m = old_ev.merge(new_ev, on=k, suffixes=("_old", "_new"))
    same_rows = len(m) == len(old_ev) == len(new_ev)
    same_auc = all(np.allclose(m[f"{c}_old"], m[f"{c}_new"], equal_nan=True)
                   for c in ["leave_auc", "join_auc"])
    old = pd.read_csv(TABLES / "rq3_summary.csv").set_index("test")
    new = pd.DataFrame([rq3._summarise(new_ev, c, N_BOOTSTRAP, 0)
                        for c in ["leave_auc", "join_auc"]]).set_index("test")
    same_sum = np.allclose(old[["AUC", "CI_low", "CI_high"]], new[["AUC", "CI_low", "CI_high"]])
    r = pd.DataFrame([dict(check="same events (terms 9-10)", rows=len(old_ev), identical=same_rows),
                      dict(check="same per-event AUCs", rows=len(m), identical=same_auc),
                      dict(check="same pooled AUC and bootstrap CI", rows=2, identical=same_sum)])
    print("\nreproduction of the notebook's saved RQ3 tables (terms 9-10):")
    print(r.to_string(index=False))
    return r


def figure(terms_table, path=None):
    """AUC per term with its cluster-bootstrap 95% CI; 0.5 = switchers vote like stayers."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    titles = {"leave": "Leavers: disloyal to their group before leaving?",
              "join": "Joiners: already voting with their future group?"}
    x = np.arange(len(TERMS))
    for ax, test in zip(axes, ["leave", "join"]):
        t = terms_table[terms_table.test == test].set_index("term").reindex(TERMS)
        ok = t.AUC.notna().to_numpy()
        ax.axhline(0.5, color=figures.INK2, lw=0.8, ls=":")
        ax.errorbar(x[ok], t.AUC[ok], yerr=[t.AUC[ok] - t.CI_low[ok], t.CI_high[ok] - t.AUC[ok]],
                    fmt="o", color=figures.GROUP_COLOURS["EPP"], capsize=3, lw=1.5, ms=6)
        for xi, (auc, n) in enumerate(zip(t.AUC, t.events)):
            if pd.notna(auc):
                ax.annotate(f"n={int(n)}", (xi, 0.02), ha="center", fontsize=7, color=figures.INK2)
        ax.set_xticks(x, [TERM_NAMES[s] for s in TERMS], fontsize=8)
        ax.set_xlim(-0.5, len(TERMS) - 0.5)
        ax.axvline(2.5, color=figures.AXIS, lw=0.8, ls="--")
        ax.set_ylim(0, 1)
        ax.set_title(titles[test], loc="left")
    axes[0].set_ylabel("AUC vs stayers (0.5 = no difference)")
    return figures._save(fig, path)


def main():
    pd.set_option("display.width", 220)
    d = load_merged(columns=["vote_id", "member_id", "position", "group_code"])
    gm = memberships()
    d["group_memberships"] = gm
    rq3.MEMBERSHIP_TERMS = term_windows()
    gone = dissolutions(gm)
    print("groups that dissolved (events left out of the main result):")
    print(gone.to_string(index=False))

    ev, per_event, summary = rq3.run(d, skip_clusters=set(gone.cluster))
    _, per_event_all, summary_all = rq3.run(d)
    terms_table = by_term(per_event)
    terms_all = by_term(per_event_all)

    ev.to_csv(TABLES / "all_terms_rq3_switches.csv", index=False)
    gone.to_csv(TABLES / "all_terms_rq3_dissolutions.csv", index=False)
    per_event.to_csv(TABLES / "all_terms_rq3_events.csv", index=False)
    summary.to_csv(TABLES / "all_terms_rq3_summary.csv", index=False)
    summary_all.to_csv(TABLES / "all_terms_rq3_summary_incl_dissolutions.csv", index=False)
    terms_table.to_csv(TABLES / "all_terms_rq3_by_term.csv", index=False)
    terms_all.to_csv(TABLES / "all_terms_rq3_by_term_incl_dissolutions.csv", index=False)
    reproduction_check(per_event).to_csv(TABLES / "all_terms_rq3_reproduction.csv", index=False)

    print("\nswitch events per term:", ev.groupby("term").size().to_dict())
    print("tested (>= enough pre-switch votes):",
          per_event[per_event.status == "ok"].groupby("term").size().to_dict())
    cols = ["test", "events", "clusters", "AUC", "CI_low", "CI_high", "share_events_above_half"]
    print("\npooled 2004-2026, dissolutions left out:")
    print(summary[cols].round(3).to_string(index=False))
    print("\npooled, all events:")
    print(summary_all[cols].round(3).to_string(index=False))
    print("\nby term (dissolutions left out):")
    print(terms_table[["term"] + cols].round(3).to_string(index=False))
    figure(terms_table, FIGS / "all_terms_rq3.png")
    print("\nsaved results/tables/all_terms_rq3_*.csv and results/figures/all_terms_rq3.png")


if __name__ == "__main__":
    main()
