"""Turns the raw HowTheyVote CSVs into three tidy parquet files in data/processed/.

Run once from the repo root:   python src/cleaned_data.py
Afterwards, load the data anywhere with:   from cleaned_data import load; d = load()

Outputs
- votes.parquet          one row per roll-call vote, with term, contested flag, passed flag
- member_votes.parquet   one row per MEP per vote (18.2M rows, compact dtypes)
- vote_topics.parquet    vote_id -> top-level OEIL policy area (a vote can have several)
"""
import pandas as pd
from paths import RAW, PROC
from config import TERMS, LOPSIDED, GROUP_ALIASES


def build():
    # --- votes ---------------------------------------------------------------------
    votes = pd.read_csv(RAW / "votes.csv.gz",
                        usecols=["id", "timestamp", "is_main", "procedure_type",
                                 "count_for", "count_against", "count_abstention", "result"])
    # Timestamps come in mixed formats (with and without fractional seconds), so
    # read_csv(parse_dates=...) silently leaves them as text. Parse explicitly.
    votes["timestamp"] = pd.to_datetime(votes["timestamp"], format="mixed")
    assert pd.api.types.is_datetime64_any_dtype(votes["timestamp"])

    votes["term"] = 0
    for term, (lo, hi) in TERMS.items():
        votes.loc[(votes.timestamp >= lo) & (votes.timestamp < hi), "term"] = term
    votes["term"] = votes["term"].astype("int8")   # 0 = outside both study periods

    # A vote is contested if the larger side got at most LOPSIDED of FOR+AGAINST votes
    cast = votes.count_for + votes.count_against
    top_share = votes[["count_for", "count_against"]].max(axis=1) / cast.where(cast > 0)
    votes["top_share"] = top_share
    votes["contested"] = top_share <= LOPSIDED

    # `result` only exists from 2024. Before that we approximate it as FOR > AGAINST.
    # This is exact for simple-majority votes, the large majority of roll calls.
    approx = (votes.count_for > votes.count_against).map({True: "ADOPTED", False: "REJECTED"})
    votes["passed"] = votes["result"].fillna(approx).eq("ADOPTED")
    votes = votes.rename(columns={"id": "vote_id"})
    votes.to_parquet(PROC / "votes.parquet", index=False)

    # --- member votes --------------------------------------------------------------
    mv = pd.read_csv(RAW / "member_votes.csv.gz",
                     usecols=["vote_id", "member_id", "position", "country_code",
                              "group_code", "national_party_id",
                              "national_party_short_label"],
                     dtype={"vote_id": "int32", "member_id": "int32",
                            "position": "category", "country_code": "category",
                            "group_code": "category", "national_party_id": "float64",
                            "national_party_short_label": "string"})
    # National party IDs change between terms (e.g. Luxembourg's DP is 5284 in term 9
    # and 6682 in term 10). "COUNTRY:label" is stable across terms and unique enough.
    mv["party"] = (mv.country_code.astype("string") + ":" +
                   mv.national_party_short_label.fillna("none")).astype("category")
    mv = mv.drop(columns=["national_party_short_label"])
    mv["group_code"] = (mv.group_code.astype("string").replace(GROUP_ALIASES)
                        .astype("category"))
    mv["national_party_id"] = mv["national_party_id"].astype("Int32")
    mv.to_parquet(PROC / "member_votes.parquet", index=False)

    # --- policy areas --------------------------------------------------------------
    sv = pd.read_csv(RAW / "oeil_subject_votes.csv.gz", dtype={"oeil_subject_code": "string"})
    sv["topic"] = sv.oeil_subject_code.str.split(".").str[0].astype(int)
    labels = pd.read_csv(RAW / "oeil_subjects.csv.gz", dtype={"code": "string"})
    labels = labels[~labels.code.str.contains(r"\.")].assign(topic=lambda d: d.code.astype(int))
    topics = (sv[["vote_id", "topic"]].drop_duplicates()
              .merge(labels[["topic", "label"]], on="topic", how="left"))
    topics.to_parquet(PROC / "vote_topics.parquet", index=False)

    print(f"votes: {len(votes):,} ({votes.term.value_counts().to_dict()})")
    print(f"member votes: {len(mv):,}")
    print(f"vote-topic links: {len(topics):,}")


def load():
    """Returns a dict of DataFrames. Builds the parquet files first if they are missing."""
    names = ["votes", "member_votes", "vote_topics"]
    if not all((PROC / f"{n}.parquet").exists() for n in names):
        build()
    d = {n: pd.read_parquet(PROC / f"{n}.parquet") for n in names}
    gm = pd.read_csv(RAW / "group_memberships.csv.gz", parse_dates=["start_date", "end_date"])
    gm["group_code"] = gm.group_code.replace(GROUP_ALIASES)
    d["group_memberships"] = gm
    d["members"] = pd.read_csv(RAW / "members.csv.gz",
                               usecols=["id", "first_name", "last_name", "country_code"])
    d["required"] = pd.read_csv(RAW / "groups_required_for_majority.csv.gz")
    return d


if __name__ == "__main__":
    build()
