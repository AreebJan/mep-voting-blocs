"""Voting power and the EPP as pivot (counting, not a course algorithm).

On votes where the left bloc and the right bloc majorities disagree: which side the EPP
majority takes, and how often each side wins with and without the EPP; and how many
votes each group could flip on its own. Used for RQ2 (why coalitions changed)."""
import numpy as np
import pandas as pd
from config import POSITIONS


LEFT_BLOC = ["SD", "GREEN_EFA", "GUE_NGL"]


RIGHT_BLOC = ["ECR", "FAR", "ESN"]           # FAR = ID (term 9) or PfE (term 10)


# --- 3. The EPP as pivot ----------------------------------------------------------------
def _group_majorities(d):
    V, mv = d["votes"], d["member_votes"]
    v = V[(V.term > 0) & V.contested]
    x = mv[mv.vote_id.isin(v.vote_id) & mv.position.isin(POSITIONS)]
    x = pd.DataFrame({"vote_id": x.vote_id.to_numpy(),
                      "group": x.group_code.astype(str).replace({"ID": "FAR", "PFE": "FAR"}).to_numpy(),
                      "pos": np.where(x.position.to_numpy() == "FOR", 1, -1)})
    net = x.groupby(["vote_id", "group"]).pos.sum().unstack(fill_value=0)
    n_for = x[x.pos == 1].groupby(["vote_id", "group"]).size().unstack(fill_value=0)
    n_against = x[x.pos == -1].groupby(["vote_id", "group"]).size().unstack(fill_value=0)
    meta = v.set_index("vote_id")[["term", "timestamp", "passed", "is_main", "procedure_type"]]
    maj = np.sign(net)
    maj["LEFT"] = np.sign(net[[g for g in LEFT_BLOC if g in net]].sum(axis=1))
    maj["RIGHT"] = np.sign(net[[g for g in RIGHT_BLOC if g in net]].sum(axis=1))
    return maj.join(meta), n_for, n_against, meta


def epp_pivot(d):
    """Votes where the left bloc and the right bloc majorities disagree: which side the
    EPP majority takes, and how often the right side wins with / without the EPP.
    Reported for all contested votes, final votes, legislation (COD), and per half-year."""
    maj, _, _, _ = _group_majorities(d)
    s = maj[(maj.LEFT != 0) & (maj.RIGHT != 0) & (maj.LEFT != maj.RIGHT) & (maj.EPP != 0)].copy()
    s["epp_with_right"] = s.EPP == s.RIGHT
    s["right_won"] = np.where(s.RIGHT == 1, s.passed, ~s.passed)
    s["half"] = s.timestamp.dt.year.astype(str) + np.where(s.timestamp.dt.month <= 6, "-H1", "-H2")

    def summarise(df):
        return pd.Series(dict(
            votes=len(df), epp_with_right=df.epp_with_right.mean(),
            right_wins_when_epp_joins=df[df.epp_with_right].right_won.mean(),
            right_wins_when_epp_against=df[~df.epp_with_right].right_won.mean()))
    subsets = {"all contested votes": s, "final votes (is_main)": s[s.is_main],
               "legislation (COD)": s[s.procedure_type == "COD"]}
    by_term = pd.concat({k: v.groupby("term").apply(summarise, include_groups=False)
                         for k, v in subsets.items()}, names=["subset"]).reset_index()
    by_half = s.groupby("half").apply(summarise, include_groups=False).reset_index()
    by_half["term"] = np.where(by_half.half >= "2024-H2", 10, 9)
    return by_term, by_half


def voting_power(d):
    """Share of contested votes whose result (FOR vs AGAINST, simple majority) would flip
    if one group's MEPs all switched sides. Simplification: abstentions and special
    majorities are ignored."""
    _, n_for, n_against, meta = _group_majorities(d)
    tot_for, tot_against = n_for.sum(axis=1), n_against.sum(axis=1)
    passed = tot_for > tot_against
    rows = []
    for g in n_for.columns:
        if g == "NI":
            continue
        f2 = tot_for - n_for[g] + n_against[g]
        a2 = tot_against - n_against[g] + n_for[g]
        flipped = (f2 > a2) != passed
        for term in (9, 10):
            ids = meta.index[meta.term == term].intersection(flipped.index)
            members = n_for.loc[ids, g] + n_against.loc[ids, g]
            if members.mean() > 0:
                rows.append(dict(group=g, term=term, pivotal_share=flipped[ids].mean(),
                                 mean_voting_members=members.mean()))
    t = pd.DataFrame(rows).pivot(index="group", columns="term",
                                 values=["pivotal_share", "mean_voting_members"])
    t.columns = [f"{name}_term{term}" for name, term in t.columns]
    return t.reset_index()
