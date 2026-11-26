"""Loading the data for the analyses, from one or more sources.

    from data_sources.dataset import load
    d = load()                     # terms 9 and 10 (HowTheyVote)
    d = load(with_votewatch=True)  # + terms 6-8 (VoteWatch, 2004-2019)
"""
import pandas as pd
from data_sources import howtheyvote


def load(with_votewatch=False):
    """Returns a dict of DataFrames. Builds the parquet files first if they are missing.

    with_votewatch=True also stacks terms 6-8 (2004-2019) from the VoteWatch export
    under the same keys, with groups re-dated per vote (see data_sources/votewatch.py).
    Those votes have term 6/7/8, vote_id >= 6,000,000, a `policy_area` column instead of
    vote_topics, and no national_party_id.
    """
    d = howtheyvote.load()
    if with_votewatch:
        from data_sources import votewatch
        vw = votewatch.load()
        for key in ["votes", "member_votes", "group_memberships", "members"]:
            d[key] = pd.concat([vw[key], d[key]], ignore_index=True)
        d["members"] = d["members"].drop_duplicates("id", keep="last")
        for c in ["position", "country_code", "group_code", "party"]:
            d["member_votes"][c] = d["member_votes"][c].astype("category")
    return d
