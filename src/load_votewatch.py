"""Loads the VoteWatch Europe data for the 2014-2019 term (EP8) into the same format as
cleaned_data.py, so the RQ1 and SVD code runs on it unchanged.

Data: VoteWatch Europe dataset (Hix, Frantescu & Hagemann), EUI research data,
https://cadmus.eui.eu/handle/1814/74918. Put these two files in data/raw/votewatch/:
    EP8_RCVs_2019_06_25.xlsx   one row per MEP, one column per vote
    EP8_Voted_docs.xlsx        one row per vote (date, result, counts, final vote?)

Run once from the repo root:   python src/load_votewatch.py
Then:   from load_votewatch import load_ep8; d8 = load_ep8()

Differences from HowTheyVote to keep in mind:
- each MEP has ONE group and ONE national party for the whole term (HowTheyVote has
  them per vote), so group switchers are labelled with a single group
- MEPs who left early have no group in the file; we fill it in from their national party
- vote codes: 1 for, 2 against, 3 abstention, 4 absent, 5 didn't vote, 0 not an MEP.
  Code 6 is not in the legend; it is treated like 4/5 (did not vote). Only FOR and
  AGAINST are used by the analysis anyway.
"""
import numpy as np
import pandas as pd
import openpyxl
from paths import RAW, PROC
from config import LOPSIDED

SRC = RAW / "votewatch"
RCV_FILE = SRC / "EP8_RCVs_2019_06_25.xlsx"
DOCS_FILE = SRC / "EP8_Voted_docs.xlsx"
TERM = 8

GROUPS = {
    "Group of the European People's Party (Christian Democrats)": "EPP",
    "Group of the Progressive Alliance of Socialists and Democrats in the European Parliament": "SD",
    "European Conservatives and Reformists Group": "ECR",
    "Group of the Alliance of Liberals and Democrats for Europe": "ALDE",
    "Group of the Greens/European Free Alliance": "GREEN_EFA",
    "Confederal Group of the European United Left - Nordic Green Left": "GUE_NGL",
    "Europe of Freedom and Direct Democracy Group": "EFDD",
    "Europe of Nations and Freedom Group": "ENF",
    "Non-attached Members": "NI",
}

# MEPs with no group whose party has no other MEP with a group in the file
MANUAL_GROUPS = {
    "GÁLL-PELCZ": "EPP",          # Fidesz (party name spelled differently)
    "JUARISTI ABAUNZ": "GUE_NGL",  # EH Bildu
    "MARAGALL": "GREEN_EFA",       # Nova Esquerra Catalana
    "MAURA BARANDIARÁN": "ALDE",   # Ciudadanos
    "PAULSEN": "ALDE",             # Folkpartiet liberalerna
    "SEBASTIÀ": "GREEN_EFA",       # Compromís
}

ISO3 = {
    "Austria": "AUT", "Belgium": "BEL", "Bulgaria": "BGR", "Croatia": "HRV", "Cyprus": "CYP",
    "Czech Republic": "CZE", "Denmark": "DNK", "Estonia": "EST", "Finland": "FIN",
    "France": "FRA", "Germany": "DEU", "Greece": "GRC", "Hungary": "HUN", "Ireland": "IRL",
    "Italy": "ITA", "Latvia": "LVA", "Lithuania": "LTU", "Luxembourg": "LUX", "Malta": "MLT",
    "Netherlands": "NLD", "Poland": "POL", "Portugal": "PRT", "Romania": "ROU",
    "Slovakia": "SVK", "Slovenia": "SVN", "Spain": "ESP", "Sweden": "SWE",
    "United Kingdom": "GBR",
}

POSITION = {1: "FOR", 2: "AGAINST", 3: "ABSTENTION"}   # everything else: did not vote


def read_rcvs():
    """MEP info and the MEP x vote code matrix."""
    wb = openpyxl.load_workbook(RCV_FILE, read_only=True)
    rows = wb["Votes"].iter_rows(values_only=True)
    header = list(next(rows))
    data = [r for r in rows if r[0] is not None]
    meps = pd.DataFrame([r[:9] for r in data], columns=header[:9])
    codes = np.array([[x if x is not None else 0 for x in r[9:]] for r in data], dtype=np.int8)
    return meps, codes, np.array(header[9:], dtype=np.int32)


def clean_meps(meps):
    meps = meps.rename(columns={"WebisteEpID": "member_id"})
    meps["group_code"] = meps.EPG.map(GROUPS)
    assert meps.group_code.notna().sum() == meps.EPG.notna().sum(), "unknown group name"

    # Fill missing groups from other MEPs of the same national party
    by_party = (meps.dropna(subset=["group_code"])
                .groupby(["Country", "Party"]).group_code.agg(lambda s: s.mode().iloc[0]))
    missing = meps.group_code.isna()
    keys = list(zip(meps.Country[missing], meps.Party[missing]))
    meps.loc[missing, "group_code"] = [by_party.get(k) for k in keys]
    missing = meps.group_code.isna()
    meps.loc[missing, "group_code"] = meps.Lname[missing].map(MANUAL_GROUPS)

    meps["country_code"] = meps.Country.map(ISO3)
    assert meps.country_code.notna().all(), "unknown country"
    meps["party"] = meps.country_code + ":" + meps.Party.fillna("none")
    return meps


def build():
    meps, codes, vote_ids = read_rcvs()
    meps = clean_meps(meps)
    no_group = meps.group_code.isna()
    print(f"{len(meps)} MEPs, {len(vote_ids)} votes; "
          f"{no_group.sum()} MEPs without a group are dropped")
    meps, codes = meps[~no_group].reset_index(drop=True), codes[~no_group.to_numpy()]

    # --- votes ---------------------------------------------------------------------
    docs = pd.read_excel(DOCS_FILE)
    docs.columns = [str(c).strip() for c in docs.columns]
    final = [c for c in docs.columns if c.startswith("Final")][0]
    votes = pd.DataFrame({
        "vote_id": docs["Vote ID"].astype("int32"),
        "timestamp": pd.to_datetime(docs["Date"]),
        "is_main": docs[final].eq(1),
        "count_for": docs["Yeas"], "count_against": docs["No"], "count_abstention": docs["Abs"],
        "passed": docs["Vote"].eq("+"),
    })
    votes["term"] = np.int8(TERM)
    cast = votes.count_for + votes.count_against
    votes["top_share"] = votes[["count_for", "count_against"]].max(axis=1) / cast.where(cast > 0)
    votes["contested"] = votes.top_share <= LOPSIDED
    votes = votes[votes.vote_id.isin(vote_ids)].reset_index(drop=True)

    # --- member votes (wide -> long) -------------------------------------------------
    r, c = np.nonzero(codes)                      # code 0 = not an MEP at the time
    code = codes[r, c]
    mv = pd.DataFrame({
        "vote_id": vote_ids[c],
        "member_id": meps.member_id.to_numpy()[r].astype("int32"),
        "position": pd.Categorical(pd.Series(code).map(POSITION).fillna("DID_NOT_VOTE")),
        "country_code": pd.Categorical(meps.country_code.to_numpy()[r]),
        "group_code": pd.Categorical(meps.group_code.to_numpy()[r]),
        "party": pd.Categorical(meps.party.to_numpy()[r]),
    })

    # Sanity check: FOR counts rebuilt from the MEP data vs the official counts
    rebuilt = mv[mv.position == "FOR"].groupby("vote_id").size()
    diff = (votes.set_index("vote_id").count_for - rebuilt.reindex(votes.vote_id).fillna(0).values).abs()
    print(f"FOR counts match the official count exactly in {(diff == 0).mean():.1%} of votes, "
          f"within 5 in {(diff <= 5).mean():.1%}")

    votes.to_parquet(PROC / "ep8_votes.parquet", index=False)
    mv.to_parquet(PROC / "ep8_member_votes.parquet", index=False)
    print(f"votes: {len(votes):,} ({votes.contested.sum():,} contested), "
          f"member votes: {len(mv):,}")


def load_ep8():
    """Same keys as cleaned_data.load() for the parts RQ1 and the SVD map need."""
    if not (PROC / "ep8_votes.parquet").exists():
        build()
    return {"votes": pd.read_parquet(PROC / "ep8_votes.parquet"),
            "member_votes": pd.read_parquet(PROC / "ep8_member_votes.parquet")}


if __name__ == "__main__":
    build()
