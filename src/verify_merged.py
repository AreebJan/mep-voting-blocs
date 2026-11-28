"""Checks the merged 2004-2026 dataset built by build_merged.py.

Run from the repo root:   python src/verify_merged.py
Prints every check and writes results/tables/merged_verification.csv (check, value,
threshold, ok). Exits with code 1 if any check fails.

What is checked
1. structure     unique (vote, MEP) rows, valid codes, no orphan rows
2. dates         weekdays only, inside each term, VoteWatch dates in vote order, no gap
                 between votes longer than an election or summer break
3. counts        official FOR/AGAINST/ABSTENTION counts vs counts rebuilt from MEP rows
4. MEP identity  every MEP is in the EP record and held a seat (a group membership) on
                 the vote date; country = EP citizenship; EP7 name matches in particular
5. groups        group per vote from the EP record vs VoteWatch's own label; no MEP-vote
                 in a group outside the group's lifetime; HowTheyVote vs EP record
6. parties       share with an EP party id; HowTheyVote party labels = EP labels; party
                 and group of MEPs sitting before and after the July 2019 seam
7. EP9 overlap   VoteWatch EP9 vs HowTheyVote on the same 2019-2022 votes: vote match
                 rate, and per-MEP positions on matched votes
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from paths import RAW, PROC
from config import EP_TERM_STARTS, EP_GROUP_CODES, HTV_START
from build_merged import load_merged, load_history, on_date, HISTORY

OUT = Path(__file__).resolve().parent.parent / "results" / "tables" / "merged_verification.csv"

# Group lifetimes (first and last day the group existed), with a few days' slack in the
# check. Groups not listed (EPP, SD, GREEN_EFA, GUE_NGL, NI) existed throughout.
GROUP_LIFETIMES = {
    "UEN": ("1999-07-20", "2009-07-13"), "IND_DEM": ("2004-07-20", "2009-07-13"),
    "ITS": ("2007-01-15", "2007-11-14"), "ALDE": ("2004-07-20", "2019-07-01"),
    "ECR": ("2009-07-14", None), "EFD": ("2009-07-14", "2014-06-30"),
    "EFDD": ("2014-07-01", "2019-07-01"), "ENF": ("2015-06-15", "2019-07-01"),
    "RENEW": ("2019-07-02", None), "ID": ("2019-07-02", "2024-07-15"),
    "PFE": ("2024-07-08", None), "ESN": ("2024-07-10", None),
}

results = []


def check(name, value, threshold, ok, detail=""):
    results.append(dict(check=name, value=value, threshold=threshold, ok=bool(ok),
                        detail=detail))
    v = f"{value:.4f}" if isinstance(value, float) else value
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {v} (want {threshold}) {detail}")


def section(title):
    print(f"\n--- {title} " + "-" * max(0, 70 - len(title)))


def structure(votes, mv):
    section("1. structure")
    dup = mv.duplicated(["vote_id", "member_id"]).sum()
    check("duplicate (vote, MEP) rows", int(dup), "0", dup == 0)
    check("duplicate vote ids", int(votes.vote_id.duplicated().sum()), "0",
          not votes.vote_id.duplicated().any())
    orphan = (~mv.vote_id.isin(votes.vote_id)).sum()
    check("MEP rows whose vote is missing", int(orphan), "0", orphan == 0)
    novoters = (~votes.vote_id.isin(mv.vote_id)).sum()
    check("votes without any MEP rows", int(novoters), "0", novoters == 0)
    bad_pos = (~mv.position.isin(["FOR", "AGAINST", "ABSTENTION", "DID_NOT_VOTE"])).sum()
    check("invalid positions", int(bad_pos), "0", bad_pos == 0)
    codes = set(pd.read_csv(RAW / "countries.csv.gz").code)
    bad_c = (~mv.country_code.isin(codes)).sum()
    check("invalid country codes", int(bad_c), "0", bad_c == 0)
    known = set(EP_GROUP_CODES.values())
    bad_g = mv.group_code.notna() & ~mv.group_code.isin(known)
    check("unknown group codes", int(bad_g.sum()), "0", not bad_g.any(),
          str(sorted(mv.group_code[bad_g].unique()))[:200])
    no_g = mv.group_code.isna().mean()
    check("MEP-votes without a group", no_g, "< 0.001", no_g < 0.001)
    terms = votes.term.value_counts().sort_index()
    check("every term 6-10 has votes", str(terms.to_dict()), "all > 1000",
          set(terms.index) == set(EP_TERM_STARTS) and (terms > 1000).all())


def dates(votes):
    section("2. dates")
    wd = votes.timestamp.dt.day_of_week
    check("votes on a weekend", int((wd >= 5).sum()), "0", (wd < 5).all())
    fri = votes[wd == 4].groupby("term").size().to_dict()
    check("votes on a Friday (info)", str(fri), "VoteWatch EP6/7 only, few", True)
    starts = pd.Series({t: pd.Timestamp(d) for t, d in EP_TERM_STARTS.items()})
    first, last = votes.groupby("term").timestamp.min(), votes.groupby("term").timestamp.max()
    ends = starts.shift(-1).fillna(pd.Timestamp("2100-01-01"))
    inside = (first >= starts) & (last < ends)
    check("first/last vote inside each term", str({t: (first[t].date().isoformat(),
          last[t].date().isoformat()) for t in first.index}), "all inside", inside.all())
    vw = votes[votes.source == "votewatch"].sort_values("vote_id")
    back = vw.groupby("term").timestamp.apply(lambda s: int((s.diff().dt.days < 0).sum()))
    check("VoteWatch dates going backwards in vote order", str(back.to_dict()), "<= 2 per term",
          (back <= 2).all())
    ts = votes.timestamp.sort_values().drop_duplicates()
    gap = ts.diff().dt.days
    big = ts[gap > 60]
    detail = ", ".join(f"{(d - pd.Timedelta(days=int(g))).date()}->{d.date()}"
                       for d, g in zip(big, gap[gap > 60]))
    check("gaps > 60 days (info)", int((gap > 60).sum()), "summer and election breaks only",
          True, detail)
    # A gap longer than a summer break (~70 days) must span an election, i.e. contain the
    # start of a term
    term_starts = [pd.Timestamp(s) for s in EP_TERM_STARTS.values()]
    long = [(d - pd.Timedelta(days=int(g)), d) for d, g in zip(ts[gap > 75], gap[gap > 75])]
    unexplained = [(a.date(), b.date()) for a, b in long
                   if not any(a < s <= b for s in term_starts)]
    check("gaps > 75 days that do not span an election", len(unexplained), "0",
          not unexplained, str(unexplained))
    seam = votes.groupby("source", observed=True).timestamp.agg(["min", "max"])
    check("VoteWatch ends before HowTheyVote starts",
          f"{seam.loc['votewatch', 'max'].date()} < {seam.loc['howtheyvote', 'min'].date()}",
          "no overlap", seam.loc["votewatch", "max"] < seam.loc["howtheyvote", "min"])


def counts(votes, mv):
    section("3. official counts vs MEP rows")
    rebuilt = mv.groupby(["vote_id", "position"], observed=True).size().unstack(fill_value=0)
    v = votes.set_index("vote_id")
    for src in ["votewatch", "howtheyvote"]:
        s = v[v.source == src]
        for pos, col in [("FOR", "count_for"), ("AGAINST", "count_against"),
                         ("ABSTENTION", "count_abstention")]:
            r = rebuilt[pos].reindex(s.index, fill_value=0)
            off = s[col].notna()
            diff = (s[col][off] - r[off]).abs()
            exact, close = (diff == 0).mean(), (diff <= 5).mean()
            check(f"{src} {pos}: official = rebuilt (exact / within 5)",
                  f"{exact:.3f} / {close:.3f}", "within 5 >= 0.95", close >= 0.95,
                  f"median |diff| {diff.median():.0f}, worst {diff.max():.0f}")


def identity(votes, mv, hist):
    section("4. MEP identity")
    vw = mv[mv.source == "votewatch"]
    ids = set(hist.member_id)
    miss = ~vw.member_id.isin(ids)
    check("VoteWatch MEPs found in the EP record", 1 - miss.mean(), ">= 0.999",
          miss.mean() <= 0.001, f"{vw.member_id[miss].nunique()} MEPs missing")
    # Held a seat = was in some EP political group on the vote date
    keys = vw[["member_id", "date", "term"]].drop_duplicates(["member_id", "date"])
    seat = on_date(keys[["member_id", "date"]], hist[hist.kind == "EU_POLITICAL_GROUP"])
    keys = keys.merge(seat[["member_id", "date"]].assign(seat=True), how="left",
                      on=["member_id", "date"])
    rate = keys.groupby("term").seat.apply(lambda s: s.notna().mean())
    check("VoteWatch MEP-days with an EP seat, by term", str(rate.round(4).to_dict()),
          ">= 0.99 each", (rate >= 0.99).all())
    per_mep = keys.groupby(["term", "member_id"]).seat.apply(lambda s: s.notna().mean())
    never = per_mep[per_mep < 0.5]
    check("VoteWatch MEPs with an EP seat on < half their vote days", len(never), "0",
          len(never) == 0, str(never.index.tolist()[:10]))
    cit = pd.read_csv(RAW / "ep_citizenship.csv").set_index("member_id").citizenship
    m = mv.drop_duplicates(["member_id", "country_code"])[["member_id", "country_code", "source"]]
    m = m.assign(cit=m.member_id.map(cit))
    bad = m[m.cit.notna() & (m.cit != m.country_code)]
    check("MEPs whose country != EP citizenship (info)", len(bad), "<= 40 (cross-border MEPs)",
          len(bad) <= 40, str(bad[["member_id", "country_code", "cit"]].values.tolist()[:15]))
    e7 = pd.read_csv(RAW / "votewatch_ep7_ids.csv")
    check("EP7 MEPs matched to an EP ID", len(e7), ">= 850 of 853", len(e7) >= 850,
          str(e7.how.value_counts().to_dict()))
    r7 = per_mep.loc[7]
    weak = e7[e7.how.ne("k_full")].member_id
    check("EP7 non-exact name matches with an EP seat on their vote days",
          float(r7.reindex(weak).min()), ">= 0.9 for every one", r7.reindex(weak).min() >= 0.9)


def groups(mv, hist):
    section("5. political groups")
    vw = mv[mv.source == "votewatch"]
    src = vw.group_source.value_counts(normalize=True).round(4).to_dict()
    check("VoteWatch MEP-votes with group from the EP record", src.get("ep_api", 0), ">= 0.99",
          src.get("ep_api", 0) >= 0.99, str(src))
    both = vw[vw.vw_group_code.notna() & vw.group_source.eq("ep_api")]
    same = both.group_code.astype(str) == both.vw_group_code.astype(str)
    agree = same.groupby(both.term).mean()
    check("dated EP group = VoteWatch's static label, by term", str(agree.round(4).to_dict()),
          ">= 0.95 each (switchers differ)", (agree >= 0.95).all())
    bad = []
    for g, (lo, hi) in GROUP_LIFETIMES.items():
        d = mv[mv.group_code == g]
        lo_ok = d.date >= pd.Timestamp(lo) - pd.Timedelta(days=3)
        hi_ok = d.date <= pd.Timestamp(hi) + pd.Timedelta(days=3) if hi else True
        n = int((~(lo_ok & hi_ok)).sum())
        if n:
            bad.append((g, n))
    check("MEP-votes in a group outside its lifetime", sum(n for _, n in bad), "0", not bad,
          str(bad))
    # The same EP-record lookup applied to HowTheyVote: does it give HowTheyVote's group?
    h = mv[mv.source == "howtheyvote"]
    keys = h[["member_id", "date"]].drop_duplicates()
    g = on_date(keys, hist[hist.kind == "EU_POLITICAL_GROUP"])
    g["api_group"] = g.org_label.map(EP_GROUP_CODES)
    m = h.merge(g[["member_id", "date", "api_group"]], on=["member_id", "date"])
    rate = (m.api_group == m.group_code).mean()
    check("HowTheyVote group = EP record group on the same day", rate, ">= 0.99", rate >= 0.99,
          f"{len(m) / len(h):.1%} of HowTheyVote MEP-votes covered by the record")


def parties(mv, hist):
    section("6. national parties")
    src = mv.groupby("source", observed=True).party_source.apply(
        lambda s: s.isin(["ep_api", "howtheyvote"]).mean())
    check("MEP-votes with a party from EP ids, by source", str(src.round(4).to_dict()),
          ">= 0.98", (src >= 0.98).all())
    no_party = mv.party.isna().mean()
    check("MEP-votes without any party label", no_party, "< 0.001", no_party < 0.001)
    # HowTheyVote's short labels vs the EP record's labels for the same party id
    lab = (hist[hist.kind == "NATIONAL_POLITICAL_GROUP"].drop_duplicates("org_id")
           .set_index("org_id").org_label)
    h = mv[mv.source == "howtheyvote"].drop_duplicates(["national_party_id", "party"]).dropna(
        subset=["national_party_id"])
    h = h.assign(api=h.country_code.astype(str) + ":" + h.national_party_id.map(lab))
    h = h[h.api.notna()]
    same = (h.api == h.party.astype(str)).mean()
    check("HowTheyVote party label = EP label for the same party id", same, ">= 0.95",
          same >= 0.95, f"{len(h)} party ids; differ: "
          f"{h[h.api != h.party.astype(str)][['party', 'api']].values.tolist()[:6]}")
    # MEPs sitting just before and just after the July 2019 seam
    seam = pd.Timestamp(HTV_START)
    pre = mv[(mv.date < seam) & (mv.date >= seam - pd.Timedelta(days=120))]
    post = mv[(mv.date >= seam) & (mv.date < seam + pd.Timedelta(days=120))]
    last = pre.sort_values("date").groupby("member_id")[["party", "group_code"]].last()
    first = post.sort_values("date").groupby("member_id")[["party", "group_code"]].first()
    both = last.join(first, lsuffix="_pre", rsuffix="_post", how="inner")
    same_p = (both.party_pre.astype(str) == both.party_post.astype(str)).mean()
    renamed = {("ALDE", "RENEW"), ("ENF", "ID"), ("EFDD", "ID"), ("EFDD", "NI")}
    g_pairs = list(zip(both.group_code_pre.astype(str), both.group_code_post.astype(str)))
    same_g = np.mean([a == b or (a, b) in renamed for a, b in g_pairs])
    diff = both[both.party_pre.astype(str) != both.party_post.astype(str)]
    check("re-elected MEPs: same party label across the 2019 seam", same_p, ">= 0.85",
          same_p >= 0.85, f"{len(both)} MEPs; differ e.g. "
          f"{diff[['party_pre', 'party_post']].values.tolist()[:6]}")
    check("re-elected MEPs: same group across the seam (ALDE->RENEW etc. allowed)", same_g,
          ">= 0.9", same_g >= 0.9)


def ep9_overlap():
    section("7. VoteWatch EP9 vs HowTheyVote (2019-2022 overlap)")
    v9 = pd.read_parquet(PROC / "vw9_votes.parquet")
    m9 = pd.read_parquet(PROC / "vw9_member_votes.parquet")
    hv = pd.read_parquet(PROC / "votes.parquet")
    hv = hv[(hv.timestamp >= v9.timestamp.min()) &
            (hv.timestamp <= v9.timestamp.max() + pd.Timedelta(days=1))]
    key = ["count_for", "count_against", "count_abstention"]
    a = v9.assign(day=v9.timestamp.dt.normalize()).dropna(subset=key)
    b = hv.assign(day=hv.timestamp.dt.normalize())
    for c in key:
        a[c], b[c] = a[c].astype(int), b[c].astype(int)
    # Same counts on the same day, or the next day (remote votes in 2020-2021 were
    # announced the day after); only unique matches are kept
    pairs = []
    for shift in (0, 1, -1):
        bb = b.assign(day=b.day - pd.Timedelta(days=shift))
        pairs.append(a.merge(bb, on=["day"] + key, suffixes=("_vw", "_htv"))
                     [["vote_id_vw", "vote_id_htv"]].assign(shift=shift))
    p = pd.concat(pairs).drop_duplicates(["vote_id_vw", "vote_id_htv"])
    p = p[~p.vote_id_vw.duplicated(keep=False) & ~p.vote_id_htv.duplicated(keep=False)]
    rate = len(p) / len(v9)
    check("EP9 VoteWatch votes matched 1:1 to a HowTheyVote vote", rate, ">= 0.80",
          rate >= 0.80, f"{len(p):,} of {len(v9):,}; HowTheyVote has {len(hv):,} in the period "
          f"(VoteWatch leaves out some votes)")
    pv = v9.set_index("vote_id").passed.reindex(p.vote_id_vw).to_numpy()
    ph = hv.set_index("vote_id").passed.reindex(p.vote_id_htv).to_numpy()
    check("matched votes: same result (passed)", float((pv == ph).mean()), ">= 0.98",
          (pv == ph).mean() >= 0.98)

    hm = pd.read_parquet(PROC / "member_votes.parquet",
                         columns=["vote_id", "member_id", "position", "group_code",
                                  "national_party_id"])
    hm = hm[hm.vote_id.isin(p.vote_id_htv)].rename(columns={"vote_id": "vote_id_htv"})
    vm = (m9[m9.vote_id.isin(p.vote_id_vw)]
          .merge(p[["vote_id_vw", "vote_id_htv"]], left_on="vote_id", right_on="vote_id_vw")
          .drop(columns=["vote_id", "vote_id_vw"]))
    j = vm.merge(hm, on=["vote_id_htv", "member_id"], how="outer", suffixes=("_vw", "_htv"),
                 indicator=True)
    only = j._merge.value_counts(normalize=True)
    check("MEP-vote rows present in both sources", float(only.get("both", 0)), ">= 0.97",
          only.get("both", 0) >= 0.97, str(only.round(4).to_dict()))
    jb = j[j._merge == "both"]
    pos_vw, pos_htv = jb.position_vw.astype(str), jb.position_htv.astype(str)
    same = (pos_vw == pos_htv).mean()
    check("same position (all four codes)", float(same), ">= 0.97", same >= 0.97)
    fa = pos_vw.isin(["FOR", "AGAINST"]) | pos_htv.isin(["FOR", "AGAINST"])
    same_fa = (pos_vw[fa] == pos_htv[fa]).mean()
    check("same position where either says FOR/AGAINST", float(same_fa), ">= 0.99",
          same_fa >= 0.99, str(pd.crosstab(pos_vw[fa], pos_htv[fa]).to_dict())[:300])
    same_g = (jb.group_code_vw.astype(str) == jb.group_code_htv.astype(str)).mean()
    check("dated EP group (built like 2004-19) = HowTheyVote group", float(same_g), ">= 0.99",
          same_g >= 0.99)
    pid = jb.dropna(subset=["national_party_id_vw", "national_party_id_htv"])
    same_p = (pid.national_party_id_vw == pid.national_party_id_htv).mean()
    check("dated EP party id (built like 2004-19) = HowTheyVote party id", float(same_p),
          ">= 0.98", same_p >= 0.98, f"{len(pid) / len(jb):.1%} of rows have both")


def main():
    d = load_merged()
    votes, mv = d["votes"], d["member_votes"]
    vi = votes.set_index("vote_id")
    mv["date"] = mv.vote_id.map(vi.timestamp).dt.normalize()
    mv["term"] = mv.vote_id.map(vi.term)
    mv["source"] = mv.vote_id.map(vi.source.astype(str))
    hist = load_history()
    print(f"merged: {len(votes):,} votes, {len(mv):,} MEP-votes; EP record: {HISTORY.name}")

    structure(votes, mv)
    dates(votes)
    counts(votes, mv)
    identity(votes, mv, hist)
    groups(mv, hist)
    parties(mv, hist)
    ep9_overlap()

    r = pd.DataFrame(results)
    r.to_csv(OUT, index=False)
    print(f"\n{r.ok.sum()} of {len(r)} checks passed; written to {OUT.relative_to(OUT.parents[2])}")
    if not r.ok.all():
        print("FAILED:", r.check[~r.ok].tolist())
        sys.exit(1)


if __name__ == "__main__":
    main()
