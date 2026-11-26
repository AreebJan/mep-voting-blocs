"""Cleans the VoteWatch Europe export (terms 6-8, 2004-2019) into the same schema as the
HowTheyVote tables, so the two can be stacked.

Run once from the repo root, after `python src/prepare_data.py group-history`:
    python src/prepare_data.py votewatch
Afterwards: `from data_sources.dataset import load; d = load(with_votewatch=True)`

The group-switcher problem
--------------------------
Each RCV workbook has ONE row per MEP with ONE political group (`EPG`), and that group is
the MEP's group when the file was exported, with no dates (`Start`/`End` are the dates of
the mandate, not of the group). So every vote an MEP cast before switching group is
credited to the group they joined later. Examples: the Front National MEPs show as ENF
from July 2014, but ENF was only founded in June 2015; in EP8, MEPs who had left by
export time have no group at all. We drop `EPG` and instead give each MEP-vote the group
the MEP belonged to *on that vote's date*, from the European Parliament's dated
membership records (data/raw/ep_group_history.csv, from data_sources/ep_api.py).

Outputs (data/processed/)
- vw_votes.parquet              one row per roll-call vote, same columns as votes.parquet
- vw_member_votes.parquet       one row per MEP per vote, same columns as member_votes.parquet
- vw_group_memberships.parquet  dated group memberships, like group_memberships.csv.gz
- vw_members.parquet            id, first_name, last_name, country_code
"""
import numpy as np
import pandas as pd
from paths import RAW, PROC
from config import (VOTEWATCH_DIR, VW_TERMS, VW_VOTE_ID_OFFSET, VW_POSITIONS,
                    EP_GROUP_CODES, VW_COUNTRY_ALIASES, LOPSIDED)

VW = RAW / VOTEWATCH_DIR
HISTORY = RAW / "ep_group_history.csv"   # written by ep_api.py
EP7_IDS = RAW / "votewatch_ep7_ids.csv"  # written by ep_api.py
META = ["id", "first_name", "last_name", "country", "party", "epg"]


def read_rcv(term):
    """Returns (meta, codes, vw_vote_ids): one meta row per MEP, an int8 MEP x vote matrix
    of VoteWatch codes, and the VoteWatch vote number of each matrix column."""
    # EP6 has a junk header row ("Column2", ...) above the real one, and a blank row below
    raw = pd.read_excel(VW / VW_TERMS[term]["rcv"], sheet_name=0,
                        header=1 if term == 6 else 0, dtype=object)
    raw = raw[raw.iloc[:, 0].notna() & (raw.iloc[:, 0].astype(str).str.strip() != "")]
    vote_cols = [c for c in raw.columns if str(c).strip().isdigit()]
    meta = pd.DataFrame({
        "id": raw.iloc[:, 0].astype(str).str.strip().astype(int).to_numpy(),
        "first_name": raw["Fname"].astype(str).str.strip().to_numpy(),
        "last_name": raw["Lname"].astype(str).str.strip().to_numpy(),
        "country": raw["Country"].astype(str).str.strip().to_numpy(),
        "party": raw["Party"].astype("string").str.strip().to_numpy(),
        "epg": raw["EPG"].astype("string").str.strip().to_numpy(),
    })
    codes = raw[vote_cols].apply(pd.to_numeric).fillna(0).to_numpy(dtype=np.int8)
    return meta, codes, np.array([int(str(c).strip()) for c in vote_cols])


def read_meta(term):
    """Only the MEP columns of an RCV workbook (much faster than read_rcv)."""
    import openpyxl
    ws = openpyxl.load_workbook(VW / VW_TERMS[term]["rcv"], read_only=True).worksheets[0]
    rows = list(ws.iter_rows(max_col=11, values_only=True))
    rows = rows[1:] if term == 6 else rows   # EP6: junk header row above the real one
    df = pd.DataFrame(rows[1:], columns=[str(c) for c in rows[0]])
    df = df[df.iloc[:, 0].notna() & (df.iloc[:, 0].astype(str).str.strip() != "")]
    return pd.DataFrame({
        "id": df.iloc[:, 0].astype(str).str.strip().astype(int).to_numpy(),
        "first_name": df["Fname"].astype(str).str.strip().to_numpy(),
        "last_name": df["Lname"].astype(str).str.strip().to_numpy(),
        "country": df["Country"].astype(str).str.strip().to_numpy(),
    })


def read_docs(term):
    """Vote metadata: VoteWatch vote number, date, final-vote flag, policy area."""
    v = pd.read_excel(VW / VW_TERMS[term]["docs"])
    if term == 6:
        cols = {"euro_act_id": "n", "date": "date", "final_vote": "final",
                "main_policy_name": "policy_area"}
    else:
        final = next(c for c in v.columns if str(c).replace("\n", "").startswith("Final"))
        area = next(c for c in v.columns if str(c) in ("De", "De/Policy area"))
        cols = {"Vote ID": "n", "Date": "date", final: "final", area: "policy_area"}
    v = v[list(cols)].rename(columns=cols)
    # Dates are mixed datetimes and "dd.mm.yyyy" text; 36 EP6 votes use the Romanian
    # month abbreviation "ian" (January).
    txt = v["date"].astype(str).str.replace(" ian ", " Jan ", regex=False)
    v["date"] = pd.to_datetime(txt, format="mixed", dayfirst=True)
    v["n"] = v["n"].astype(int)
    v["final"] = pd.to_numeric(v["final"], errors="coerce").fillna(0).astype(bool)
    return v


def ep_ids(term, meta):
    """EP website IDs (= HowTheyVote member_id). EP6 and EP8 already use them; EP7 uses
    its own row number, mapped by name in ep_api.py."""
    if term != 7:
        return meta["id"]
    ids = pd.read_csv(EP7_IDS).set_index("vw_id")["member_id"]
    return meta["id"].map(ids)


def load_history():
    """Dated political-group memberships with HowTheyVote-style group codes."""
    h = pd.read_csv(HISTORY, parse_dates=["start_date", "end_date"])
    h = h[h.kind == "EU_POLITICAL_GROUP"].copy()
    unknown = set(h.org_label) - set(EP_GROUP_CODES)
    assert not unknown, f"add these EP group abbreviations to EP_GROUP_CODES: {unknown}"
    h["group_code"] = h.org_label.map(EP_GROUP_CODES)
    return h[["member_id", "group_code", "start_date", "end_date"]]


def group_on_date(mv, history):
    """Adds group_code: the group each MEP belonged to on each vote's date. On a switch
    day both memberships can cover the date; the one that started later wins."""
    keys = mv[["member_id", "date"]].drop_duplicates()
    m = keys.merge(history, on="member_id")
    m = m[(m.start_date <= m.date) & (m.end_date.isna() | (m.date <= m.end_date))]
    m = m.sort_values("start_date").drop_duplicates(["member_id", "date"], keep="last")
    return mv.merge(m[["member_id", "date", "group_code"]], on=["member_id", "date"],
                    how="left")


def build_term(term, history, countries):
    meta, codes, ns = read_rcv(term)
    meta["member_id"] = ep_ids(term, meta)
    missing = meta.member_id.isna()
    if missing.any():
        print(f"  EP{term}: {missing.sum()} MEPs without an EP ID are dropped: "
              f"{(meta.last_name[missing]).tolist()}")
    docs = read_docs(term).set_index("n").loc[ns].reset_index()
    docs["vote_id"] = term * VW_VOTE_ID_OFFSET + docs["n"]

    # --- long format: one row per MEP per vote they were an MEP for ------------------
    rows, cols = np.nonzero((codes > 0) & ~missing.to_numpy()[:, None])
    mv = pd.DataFrame({
        "vote_id": docs.vote_id.to_numpy()[cols].astype("int32"),
        "member_id": meta.member_id.to_numpy()[rows].astype("int32"),
        "position": pd.Series(codes[rows, cols]).map(VW_POSITIONS).to_numpy(),
        "date": docs.date.to_numpy()[cols],
        "vw_epg": meta.epg.to_numpy()[rows],
    })
    country = meta.country.replace(VW_COUNTRY_ALIASES).map(countries)
    assert country.notna().all(), set(meta.country[country.isna()])
    mv["country_code"] = country.to_numpy()[rows]
    mv["party"] = (country + ":" + meta.party.fillna("none")).to_numpy()[rows]
    mv = group_on_date(mv, history)

    # --- votes table, counts recomputed from the member rows ---------------------------
    cnt = mv.groupby(["vote_id", "position"]).size().unstack(fill_value=0)
    votes = docs[["vote_id", "date", "final", "policy_area"]].rename(
        columns={"date": "timestamp", "final": "is_main"}).set_index("vote_id")
    for pos, col in [("FOR", "count_for"), ("AGAINST", "count_against"),
                     ("ABSTENTION", "count_abstention")]:
        votes[col] = cnt.get(pos, pd.Series(dtype=int)).reindex(votes.index, fill_value=0)
    votes = votes.reset_index()
    votes["term"] = np.int8(term)
    cast = votes.count_for + votes.count_against
    votes["top_share"] = (votes[["count_for", "count_against"]].max(axis=1)
                          / cast.where(cast > 0))
    votes["contested"] = votes.top_share <= LOPSIDED
    # Same approximation howtheyvote.build uses before 2024: passed = FOR > AGAINST
    votes["passed"] = votes.count_for > votes.count_against

    report(term, mv, meta)
    members = meta.assign(country_code=country)[
        ["member_id", "first_name", "last_name", "country_code"]].rename(
        columns={"member_id": "id"})
    return votes, mv.drop(columns=["date", "vw_epg"]), members


def report(term, mv, meta):
    """How many MEP-votes the dated groups move, compared with VoteWatch's static label."""
    no_group = mv.group_code.isna()
    vw_code = mv.vw_epg.map(VW_EPG_CODES)
    moved = mv.group_code.notna() & vw_code.notna() & (mv.group_code != vw_code)
    per_mep = mv[moved].groupby("member_id").size()
    print(f"  EP{term}: {len(mv):,} MEP-votes | no dated group: {no_group.sum():,} | "
          f"group differs from VoteWatch's static label: {moved.sum():,} "
          f"({moved.mean():.1%}) across {len(per_mep)} MEPs | "
          f"VoteWatch label missing: {vw_code.isna().sum():,}")


# VoteWatch's long group names, only used to report how much the dated groups change
VW_EPG_CODES = {
    "Group of the European People's Party (Christian Democrats) and European Democrats": "EPP",
    "Group of the European People's Party (Christian Democrats)": "EPP",
    "Socialist Group in the European Parliament": "SD",
    "Group of the Progressive Alliance of Socialists and Democrats in the European Parliament": "SD",
    "Group of the Alliance of Liberals and Democrats for Europe": "ALDE",
    "Union for Europe of the Nations Group": "UEN",
    "Confederal Group of the European United Left - Nordic Green Left": "GUE_NGL",
    "Group of the Greens/European Free Alliance": "GREEN_EFA",
    "Non-attached Members": "NI",
    "Independence/Democracy Group": "IND_DEM",
    "European Conservatives and Reformists Group": "ECR",
    "Europe of freedom and democracy Group": "EFD",
    "Europe of Freedom and Direct Democracy Group": "EFDD",
    "Europe of Nations and Freedom Group": "ENF",
}


def build():
    history = load_history()
    countries = pd.read_csv(RAW / "countries.csv.gz").set_index("label")["code"]
    out = [build_term(t, history, countries) for t in VW_TERMS]
    votes = pd.concat([o[0] for o in out], ignore_index=True)
    mv = pd.concat([o[1] for o in out], ignore_index=True)
    for c in ["position", "country_code", "group_code", "party"]:
        mv[c] = mv[c].astype("category")
    members = pd.concat([o[2] for o in out]).drop_duplicates("id", keep="last")

    # Group memberships in HowTheyVote's format, clipped to the VoteWatch terms
    gm = []
    for t, spec in VW_TERMS.items():
        lo, hi = map(pd.Timestamp, spec["dates"])
        g = history[(history.start_date < hi) &
                    (history.end_date.isna() | (history.end_date >= lo))].copy()
        g = g[g.member_id.isin(mv.member_id[mv.vote_id // VW_VOTE_ID_OFFSET == t])]
        g["term"] = t
        gm.append(g)
    gm = pd.concat(gm, ignore_index=True)[
        ["member_id", "group_code", "term", "start_date", "end_date"]]

    votes.to_parquet(PROC / "vw_votes.parquet", index=False)
    mv.to_parquet(PROC / "vw_member_votes.parquet", index=False)
    gm.to_parquet(PROC / "vw_group_memberships.parquet", index=False)
    members.to_parquet(PROC / "vw_members.parquet", index=False)
    print(f"vw votes: {len(votes):,} ({votes.term.value_counts().sort_index().to_dict()})")
    print(f"vw member votes: {len(mv):,}")


NAMES = ["vw_votes", "vw_member_votes", "vw_group_memberships", "vw_members"]


def load():
    """Returns a dict with keys votes, member_votes, group_memberships, members."""
    if not all((PROC / f"{n}.parquet").exists() for n in NAMES):
        build()
    return {n[3:]: pd.read_parquet(PROC / f"{n}.parquet") for n in NAMES}
