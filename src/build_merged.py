"""Merged roll-call dataset 2004-2026: VoteWatch Europe (EP6-EP8, 2004-2019) stacked with
HowTheyVote (Jul 2019 - now), with every MEP's political group and national party taken
from the European Parliament's dated membership records for the day of each vote.

Run from the repo root, after download_data.py, cleaned_data.py and fetch_ep_history.py:
    python src/build_merged.py       (~5 min, ~6 GB RAM)
    python src/verify_merged.py      checks the result
Load:   from build_merged import load_merged; d = load_merged()

Outputs (data/processed/)
- merged_votes.parquet          one row per roll-call vote
    vote_id          HowTheyVote id, or term * 1,000,000 + VoteWatch vote number
    source           "votewatch" | "howtheyvote";   source_vote_id: id in that source
    term             6-10 (EP_TERM_STARTS: term 9 starts July 2019, before Brexit)
    timestamp, title, is_main, policy_area (VoteWatch only), procedure_type (COD, INI, ...;
    sparse in EP6), count_for/against/abstention
    (official counts; counts_source = "rebuilt" where VoteWatch's are missing or
    impossible), passed, top_share, contested (top_share <= LOPSIDED)
- merged_member_votes.parquet   one row per MEP per vote they were an MEP for (VoteWatch
  "absent" rows outside the MEP's EP seat are dropped: VoteWatch fills them in for
  periods before a substitute took the seat)
    vote_id, member_id (EP ID, the same in both sources), position (FOR, AGAINST,
    ABSTENTION, DID_NOT_VOTE), country_code (ISO-3), group_code, national_party_id (EP
    organisation id), party ("COUNTRY:short label"), group_source / party_source
    ("ep_api" = dated EP record, "votewatch" = VoteWatch's single end-of-term label used
    as a fallback, "howtheyvote" = HowTheyVote's own per-vote value)
- vw9_votes.parquet, vw9_member_votes.parquet   VoteWatch EP9 (2019-2022), built the same
  way but NOT in the merged data: it duplicates HowTheyVote and only serves verification
"""
import datetime as dt
import numpy as np
import pandas as pd
import openpyxl
from paths import RAW, PROC
from config import (VOTEWATCH_DIR, VW_TERMS, MERGED_VW_TERMS, HTV_START, EP_TERM_STARTS,
                    VW_VOTE_ID_OFFSET, VW_POSITIONS, EP_GROUP_CODES, VW_COUNTRY_ALIASES,
                    LOPSIDED, GROUP_ALIASES)

VW = RAW / VOTEWATCH_DIR
HISTORY = RAW / "ep_group_history.csv"     # fetch_ep_history.py
EP7_IDS = RAW / "votewatch_ep7_ids.csv"    # fetch_ep_history.py
OUT = {"votes": PROC / "merged_votes.parquet", "member_votes": PROC / "merged_member_votes.parquet"}

# VoteWatch's long group names -> codes. Only a fallback for MEP-votes that the EP
# record does not cover, and for the comparison in verify_merged.py.
VW_EPG_CODES = {
    "Group of the European People's Party (Christian Democrats) and European Democrats": "EPP",
    "Group of the European People's Party (Christian Democrats)": "EPP",
    "Socialist Group in the European Parliament": "SD",
    "Group of the Progressive Alliance of Socialists and Democrats in the European Parliament": "SD",
    "Group of the Alliance of Liberals and Democrats for Europe": "ALDE",
    "Renew Europe Group": "RENEW",
    "Union for Europe of the Nations Group": "UEN",
    "Confederal Group of the European United Left - Nordic Green Left": "GUE_NGL",
    "The Left group in the European Parliament - GUE/NGL": "GUE_NGL",
    "Group of the Greens/European Free Alliance": "GREEN_EFA",
    "Non-attached Members": "NI",
    "Independence/Democracy Group": "IND_DEM",
    "Identity, Tradition and Sovereignty Group": "ITS",
    "European Conservatives and Reformists Group": "ECR",
    "Europe of freedom and democracy Group": "EFD",
    "Europe of Freedom and Direct Democracy Group": "EFDD",
    "Europe of Nations and Freedom Group": "ENF",
    "Identity and Democracy Group": "ID",
}


# --- VoteWatch -----------------------------------------------------------------------
def read_rcv(term):
    """MEP columns and the MEP x vote matrix of VoteWatch codes (int8, 0 = not an MEP)."""
    ws = openpyxl.load_workbook(VW / VW_TERMS[term]["rcv"], read_only=True).worksheets[0]
    rows = ws.iter_rows(values_only=True)
    if term == 6:
        next(rows)                            # EP6: junk header row above the real one
    header = [str(c).strip() if c is not None else "" for c in next(rows)]
    vote_idx = [i for i, h in enumerate(header) if h.isdigit()]
    meta, codes = [], []
    for r in rows:
        if r[0] is None or str(r[0]).strip() == "":
            continue
        meta.append(r[:vote_idx[0]])
        codes.append([r[i] if i < len(r) else None for i in vote_idx])
    meta = pd.DataFrame(meta, columns=header[:vote_idx[0]])
    codes = pd.DataFrame(codes).apply(pd.to_numeric, errors="coerce").fillna(0)
    meta = pd.DataFrame({
        "vw_id": meta.iloc[:, 0].astype(str).str.strip().astype(int),
        "first_name": meta["Fname"].astype(str).str.strip(),
        "last_name": meta["Lname"].astype(str).str.strip(),
        "country": meta["Country"].astype(str).str.strip(),
        "vw_party": meta["Party"].astype("string").str.strip(),
        "vw_epg": meta["EPG"].astype("string").str.strip(),
    })
    return meta, codes.to_numpy(dtype=np.int8), np.array([int(header[i]) for i in vote_idx])


def _col(df, *names):
    """First column whose name (whitespace collapsed) is one of names."""
    clean = {" ".join(str(c).split()): c for c in df.columns}
    for n in names:
        if n in clean:
            return df[clean[n]]
    raise KeyError(f"none of {names} in {list(clean)}")


def parse_date(x):
    """VoteWatch date cells are a mix of Excel dates (correct as stored), "dd.mm.yyyy"
    text, "dd ian yyyy" (Romanian January, EP6) and one US-style "3/13/2008" (EP6)."""
    if isinstance(x, (dt.datetime, pd.Timestamp)):
        return pd.Timestamp(x).normalize()
    s = str(x).strip().replace(" ian ", " Jan ")
    if "/" in s:
        return pd.to_datetime(s, format="%m/%d/%Y")
    return pd.to_datetime(s, dayfirst=True)


def repair_dates(dates, suspect_days=60, window=6):
    """VoteWatch numbers votes in chronological order, so a run of same-date votes that
    is out of order is a typo. A run is suspect if it is more than suspect_days from the
    median date of the window runs on each side; a suspect run that also breaks the order
    (earlier than the last good run before it or later than the first good run after it)
    is moved only if one simple fix (day and month swapped, or the year off by 1-2) puts
    it on a weekday between those two good runs. Returns (dates, fixes)."""
    d = dates.reset_index(drop=True).copy()
    run = (d != d.shift()).cumsum()
    runs = d.groupby(run).first()
    ks = list(runs.index)
    suspect = {}
    for i, k in enumerate(ks):
        around = runs.iloc[max(0, i - window):i].tolist() + runs.iloc[i + 1:i + 1 + window].tolist()
        ref = sorted(around)[len(around) // 2]
        suspect[k] = abs((runs[k] - ref).days) > suspect_days
    fixes = []
    for i, k in enumerate(ks):
        if not suspect[k]:
            continue
        left = [runs[j] for j in ks[max(0, i - window):i] if not suspect[j]]
        right = [runs[j] for j in ks[i + 1:i + 1 + window] if not suspect[j]]
        lo = max(left) if left else pd.Timestamp.min
        hi = min(right) if right else pd.Timestamp.max
        x = runs[k]
        if lo <= x <= hi:
            continue
        cands = [x.replace(year=x.year + y) for y in (-2, -1, 1, 2)]
        if x.day <= 12:
            cands.append(pd.Timestamp(x.year, x.day, x.month))
        ok = [c for c in cands if c.day_of_week < 5 and lo <= c <= hi]
        if len(ok) == 1:
            runs[k] = ok[0]
            fixes.append((x.date(), ok[0].date(), int((run == k).sum())))
        else:
            print(f"  date {x.date()} out of order ({int((run == k).sum())} votes), left as is")
    return run.map(runs).to_numpy(), fixes


# OEIL procedure types (HowTheyVote's procedure_type values plus older ones)
PROCEDURE_TYPES = ["COD", "CNS", "INI", "RSP", "BUD", "NLE", "INL", "DEC", "APP", "AVC", "SYN",
                   "BUI", "REG", "RSO", "DEA", "ACI", "RPS", "INS", "NOM", "DCE", "IMM", "ITS"]


def clean_policy_area(s):
    """VoteWatch spells its policy areas inconsistently ("Foreign& security policy",
    "Budgetary Control", "Juridical affairs" = "Legal affairs", EP6's "Economics" =
    "Economic & monetary affairs"). One spelling per area, sentence case."""
    s = (s.astype("string").str.strip().str.lower()
         .str.replace(r"\s*(&|\band\b)\s*", " & ", regex=True)
         .str.replace(r"\s+", " ", regex=True))
    s = s.replace({"juridical affairs": "legal affairs",
                   "economics": "economic & monetary affairs",
                   "regioanal development": "regional development"})
    return (s.str[:1].str.upper() + s.str[1:]).str.replace(r"\bep\b", "EP", regex=True)


def read_docs(term):
    v = pd.read_excel(VW / VW_TERMS[term]["docs"])
    docs = pd.DataFrame({
        "n": _col(v, "euro_act_id", "Vote ID").astype(int),
        "raw_date": _col(v, "date", "Date"),
        "is_main": pd.to_numeric(_col(v, "final_vote", "Final vote?"), errors="coerce")
                   .fillna(0).astype(bool),
        "policy_area": clean_policy_area(
            _col(v, "main_policy_name", "De", "De/Policy area", "Policy area")),
        "title": _col(v, "title", "Title").astype("string").str.strip(),
        "result": _col(v, "result_code", "Vote").astype("string").str.strip(),
        "count_for": pd.to_numeric(_col(v, "yes", "Yeas", "Yes"), errors="coerce"),
        "count_against": pd.to_numeric(_col(v, "no", "No"), errors="coerce"),
        "count_abstention": pd.to_numeric(_col(v, "abstain", "Abs"), errors="coerce"),
        # OEIL procedure type (COD, INI, RSP, ...): in EP7-EP9 it ends the
        # interinstitutional file number ("2014/0170/NLE"); EP6 has it in `procedure` for
        # only ~21% of votes, so EP6 is left out wherever procedure_type matters
        "procedure_type": _col(v, "interinstitutional file number", "Interinstitutional file number",
                               "procedure")
                          .astype("string").str.upper()
                          .str.extract(rf"\b({'|'.join(PROCEDURE_TYPES)})\b")[0],
    }).sort_values("n").reset_index(drop=True)
    dates = docs.raw_date.map(parse_date)
    docs["timestamp"], fixes = repair_dates(dates)
    for old, new, n in fixes:
        print(f"  EP{term}: {n} votes dated {old} moved to {new}")
    return docs.drop(columns="raw_date")


def load_history():
    h = pd.read_csv(HISTORY, parse_dates=["start_date", "end_date"])
    h["org_id"] = h.org_id.astype(int)
    return h


def on_date(keys, h):
    """For each (member_id, date) in keys: the membership in h covering that date. On a
    switch day two can cover it; the one that started later wins."""
    m = keys.merge(h, on="member_id")
    m = m[(m.start_date <= m.date) & (m.end_date.isna() | (m.date <= m.end_date))]
    return m.sort_values("start_date").drop_duplicates(["member_id", "date"], keep="last")


def add_group_and_party(mv, history, vw_epg=None, vw_party=None):
    """group_code / national_party_id / party for each MEP-vote from the EP record on the
    vote's date. Where the record has nothing, VoteWatch's static label (if given)."""
    keys = mv[["member_id", "date"]].drop_duplicates()
    g = on_date(keys, history[history.kind == "EU_POLITICAL_GROUP"])
    unknown = set(g.org_label) - set(EP_GROUP_CODES)
    assert not unknown, f"add these EP group labels to config.EP_GROUP_CODES: {unknown}"
    g["group_code"] = g.org_label.map(EP_GROUP_CODES)
    p = on_date(keys, history[history.kind == "NATIONAL_POLITICAL_GROUP"])
    p = p.rename(columns={"org_id": "national_party_id", "org_label": "party_label"})
    mv = (mv.merge(g[["member_id", "date", "group_code"]], on=["member_id", "date"], how="left")
            .merge(p[["member_id", "date", "national_party_id", "party_label"]],
                   on=["member_id", "date"], how="left"))
    mv["group_source"] = np.where(mv.group_code.notna(), "ep_api", None)
    mv["party_source"] = np.where(mv.national_party_id.notna(), "ep_api", None)
    mv["party"] = mv.country_code + ":" + mv.party_label
    if vw_epg is not None:
        fb = mv.group_code.isna() & vw_epg.map(VW_EPG_CODES).notna()
        mv.loc[fb, "group_code"] = vw_epg[fb].map(VW_EPG_CODES)
        mv.loc[fb, "group_source"] = "votewatch"
    if vw_party is not None:
        fb = mv.party.isna() & vw_party.notna()
        mv.loc[fb, "party"] = mv.country_code[fb] + ":" + vw_party[fb]
        mv.loc[fb, "party_source"] = "votewatch"
    return mv.drop(columns="party_label")


def build_vw_term(term, history, countries):
    print(f"EP{term}: reading VoteWatch ...")
    meta, codes, ns = read_rcv(term)
    if term == 7:     # EP7 numbers its MEPs 1..853; EP IDs matched by name
        ids = pd.read_csv(EP7_IDS).set_index("vw_id").member_id
        meta["member_id"] = meta.vw_id.map(ids)
    else:
        meta["member_id"] = meta.vw_id
    missing = meta.member_id.isna()
    if missing.any():
        print(f"  {missing.sum()} MEPs without an EP ID dropped: {meta.last_name[missing].tolist()}")
    meta["country_code"] = meta.country.replace(VW_COUNTRY_ALIASES).map(countries)
    assert meta.country_code.notna().all(), set(meta.country[meta.country_code.isna()])

    docs = read_docs(term).set_index("n").loc[ns].reset_index()
    docs["vote_id"] = term * VW_VOTE_ID_OFFSET + docs.n

    r, c = np.nonzero((codes > 0) & ~missing.to_numpy()[:, None])
    take = lambda col: meta[col].to_numpy()[r]
    mv = pd.DataFrame({
        "vote_id": docs.vote_id.to_numpy()[c].astype("int32"),
        "member_id": take("member_id").astype("int32"),
        "position": pd.Series(codes[r, c]).map(VW_POSITIONS).to_numpy(),
        "date": docs.timestamp.to_numpy()[c],
        "country_code": take("country_code"),
    })
    mv = add_group_and_party(mv, history, vw_epg=pd.Series(take("vw_epg")),
                             vw_party=pd.Series(take("vw_party")))
    mv["vw_group_code"] = pd.Series(take("vw_epg")).map(VW_EPG_CODES).to_numpy()

    # VoteWatch marks people as absent (code 4/5) for stretches when they were not MEPs,
    # e.g. before a substitute took their seat. Where the EP record has no seat on that
    # day and the row is DID_NOT_VOTE, it is not a real MEP-vote: drop it. Rows where
    # they did vote are kept (with VoteWatch's group label): a cast vote proves the seat.
    ghost = mv.group_source.ne("ep_api") & mv.position.eq("DID_NOT_VOTE")
    print(f"  {ghost.sum():,} DID_NOT_VOTE rows outside the MEP's EP seat dropped; "
          f"{(mv.group_source.ne('ep_api') & ~ghost).sum():,} cast votes kept without an EP seat")
    mv = mv[~ghost].reset_index(drop=True)

    # Official counts; rebuilt from the MEP rows where they are missing or impossible
    # (one EP6 vote has FOR = 397,129,155 and no AGAINST/ABSTENTION)
    cols = {"FOR": "count_for", "AGAINST": "count_against", "ABSTENTION": "count_abstention"}
    rebuilt = (mv.groupby(["vote_id", "position"]).size().unstack(fill_value=0)
               .reindex(index=docs.vote_id, columns=list(cols), fill_value=0))
    n_meps = mv.groupby("vote_id").size().reindex(docs.vote_id, fill_value=0).to_numpy()
    official = docs[list(cols.values())]
    bad = (official.isna().any(axis=1) | (official.sum(axis=1) > n_meps)).to_numpy()
    for pos, col in cols.items():
        docs[col] = np.where(bad, rebuilt[pos].to_numpy(), docs[col])
    print(f"  {bad.sum()} votes with missing or impossible official counts: rebuilt from MEP rows")

    passed = docs.result.map({"+": True, "-": False, "-*": False})
    votes = pd.DataFrame({
        "vote_id": docs.vote_id.astype("int32"), "source": "votewatch",
        "source_vote_id": docs.n.astype("int32"), "timestamp": docs.timestamp,
        "title": docs.title, "is_main": docs.is_main, "policy_area": docs.policy_area,
        "procedure_type": docs.procedure_type,
        "count_for": docs.count_for, "count_against": docs.count_against,
        "count_abstention": docs.count_abstention,
        "counts_source": np.where(bad, "rebuilt", "official"),
        "passed": passed.fillna(docs.count_for > docs.count_against).astype(bool),
    })
    print(f"  {len(votes):,} votes, {len(mv):,} MEP-votes, "
          f"group from EP record {mv.group_source.eq('ep_api').mean():.1%}, "
          f"party from EP record {mv.party_source.eq('ep_api').mean():.1%}")
    return votes, mv.drop(columns="date")


# --- HowTheyVote ---------------------------------------------------------------------
def build_htv():
    """HowTheyVote's per-vote group and national party are kept as they are: they come
    from the same EP records (verify_merged.py checks this)."""
    v = pd.read_parquet(PROC / "votes.parquet")
    v = v[v.timestamp >= HTV_START]
    titles = pd.read_csv(RAW / "votes.csv.gz", usecols=["id", "display_title"]) \
               .set_index("id").display_title
    votes = pd.DataFrame({
        "vote_id": v.vote_id.astype("int32"), "source": "howtheyvote",
        "source_vote_id": v.vote_id.astype("int32"), "timestamp": v.timestamp,
        "title": v.vote_id.map(titles).astype("string"), "is_main": v.is_main.astype(bool),
        "procedure_type": v.procedure_type.astype("string"),
        "policy_area": pd.Series(pd.NA, index=v.index, dtype="string"),
        "count_for": v.count_for, "count_against": v.count_against,
        "count_abstention": v.count_abstention, "counts_source": "official",
        "passed": v.passed.astype(bool),
    })
    mv = pd.read_parquet(PROC / "member_votes.parquet")
    mv = mv[mv.vote_id.isin(votes.vote_id)]
    mv = pd.DataFrame({
        "vote_id": mv.vote_id.to_numpy(), "member_id": mv.member_id.to_numpy(),
        "position": mv.position.astype(str).to_numpy(),
        "country_code": mv.country_code.astype(str).to_numpy(),
        "group_code": mv.group_code.astype(str).replace(GROUP_ALIASES).to_numpy(),
        "national_party_id": mv.national_party_id.to_numpy(),
        "party": mv.party.astype(str).to_numpy(),
        "group_source": "howtheyvote", "party_source": "howtheyvote",
    })
    return votes, mv


def finish_votes(votes):
    starts = pd.Series({t: pd.Timestamp(d) for t, d in EP_TERM_STARTS.items()}).sort_values()
    votes["term"] = pd.cut(votes.timestamp, list(starts) + [pd.Timestamp("2100-01-01")],
                           labels=list(starts.index), right=False).astype("int8")
    cast = votes.count_for + votes.count_against
    votes["top_share"] = votes[["count_for", "count_against"]].max(axis=1) / cast.where(cast > 0)
    votes["contested"] = votes.top_share <= LOPSIDED
    for c in ["count_for", "count_against", "count_abstention"]:
        votes[c] = votes[c].astype("Int32")
    votes["source"] = votes.source.astype("category")
    votes["counts_source"] = votes.counts_source.astype("category")
    return votes.sort_values(["timestamp", "vote_id"]).reset_index(drop=True)


def compact(mv):
    for c in ["position", "country_code", "group_code", "party", "group_source", "party_source",
              "vw_group_code"]:
        mv[c] = mv[c].astype("category")
    mv["national_party_id"] = mv.national_party_id.astype("Int32")
    return mv


def build():
    history = load_history()
    countries = pd.read_csv(RAW / "countries.csv.gz").set_index("label")["code"]
    vw = {t: build_vw_term(t, history, countries) for t in VW_TERMS}

    # VoteWatch EP9: verification only
    v9, mv9 = vw.pop(9)
    finish_votes(v9).to_parquet(PROC / "vw9_votes.parquet", index=False)
    compact(mv9).to_parquet(PROC / "vw9_member_votes.parquet", index=False)

    htv_votes, htv_mv = build_htv()
    votes = finish_votes(pd.concat([vw[t][0] for t in MERGED_VW_TERMS] + [htv_votes],
                                   ignore_index=True))
    mv = pd.concat([vw[t][1] for t in MERGED_VW_TERMS] + [htv_mv], ignore_index=True)
    mv = compact(mv)
    votes.to_parquet(OUT["votes"], index=False)
    mv.to_parquet(OUT["member_votes"], index=False)
    print(f"merged: {len(votes):,} votes {votes.timestamp.min().date()} - "
          f"{votes.timestamp.max().date()} ({votes.term.value_counts().sort_index().to_dict()}), "
          f"{len(mv):,} MEP-votes, {mv.member_id.nunique():,} MEPs")


def load_merged(columns=None):
    """{"votes": ..., "member_votes": ...}; builds the files first if they are missing.
    vw_group_code (VoteWatch's static label) is only kept for verification; drop it
    with columns=[...] if memory is tight."""
    if not all(f.exists() for f in OUT.values()):
        build()
    return {"votes": pd.read_parquet(OUT["votes"]),
            "member_votes": pd.read_parquet(OUT["member_votes"], columns=columns)}


if __name__ == "__main__":
    build()
