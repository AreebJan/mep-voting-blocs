"""MEP x vote matrices, pairwise agreement and the kNN agreement graph.

Everything is computed with sparse matrix products, so a whole term (about 18,000 votes,
6M FOR/AGAINST records) takes seconds instead of a dense pandas pivot.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from config import TERMS, POSITIONS, MIN_VOTES, MIN_SHARED, K


def term_records(d, term, vote_ids=None, min_votes=MIN_VOTES):
    """FOR/AGAINST records on contested votes of one term, for MEPs with >= min_votes.

    vote_ids: optionally restrict to these votes (e.g. one policy area)."""
    votes = d["votes"]
    v = votes[(votes.term == term) & votes.contested]
    if vote_ids is not None:
        v = v[v.vote_id.isin(vote_ids)]
    mv = d["member_votes"]
    mv = mv[mv.vote_id.isin(v.vote_id) & mv.position.isin(POSITIONS)]
    n = mv.groupby("member_id", observed=True).size()
    mv = mv[mv.member_id.isin(n[n >= min_votes].index)]
    return mv


def member_labels(mv):
    """One label per MEP: the group / party held on most of their votes, and country."""
    def mode(s):
        return s.astype("string").fillna("none").mode().iloc[0]
    lab = mv.groupby("member_id", observed=True).agg(
        group=("group_code", mode), country=("country_code", mode), party=("party", mode),
        n_votes=("vote_id", "size"))
    return lab


def vote_matrix(mv, members, vote_ids=None):
    """Sparse MEP x vote matrix: +1 FOR, -1 AGAINST, 0 missing.
    Rows follow `members`, columns follow `vote_ids` (sorted unique by default)."""
    if vote_ids is None:
        vote_ids = np.sort(mv.vote_id.unique())
    mi = pd.Series(np.arange(len(members)), index=members)
    vi = pd.Series(np.arange(len(vote_ids)), index=vote_ids)
    r = mi.reindex(mv.member_id).to_numpy()
    c = vi.reindex(mv.vote_id).to_numpy()
    ok = ~(np.isnan(r) | np.isnan(c))
    val = np.where(mv.position.to_numpy() == "FOR", 1.0, -1.0)
    X = sp.csr_matrix((val[ok], (r[ok].astype(int), c[ok].astype(int))),
                      shape=(len(members), len(vote_ids)))
    return X, np.asarray(vote_ids)


def agreement(X, min_shared=MIN_SHARED):
    """Share of co-attended votes on which two MEPs took the same side.

    With x in {+1,-1}: same side <=> x_i*x_j = +1, so
        agreements = (shared + X X^T) / 2   and   agreement = agreements / shared.
    Pairs with fewer than min_shared shared votes get NaN."""
    O = abs(X)                                   # 1 where the MEP voted
    shared = (O @ O.T).toarray()
    prod = (X @ X.T).toarray()
    agree = np.full(shared.shape, np.nan)
    ok = shared >= min_shared
    agree[ok] = (shared[ok] + prod[ok]) / (2 * shared[ok])
    np.fill_diagonal(agree, np.nan)
    return agree, shared


def knn_graph(agree, k=K):
    """Symmetric kNN backbone: each MEP keeps edges to its k most-agreeing colleagues
    (weight = agreement); an edge exists if either endpoint chose it.

    Ties are broken by row order with a stable sort, so the graph is identical on every
    machine. (In the 300-vote prototype, 140 of 706 MEPs had a tie at the 15th place.)"""
    n = len(agree)
    a = np.nan_to_num(agree, nan=-1.0)
    W = np.zeros((n, n))
    for i in range(n):
        order = np.argsort(-a[i], kind="stable")
        best = [j for j in order[:k + 1] if j != i and a[i, j] >= 0][:k]
        W[i, best] = agree[i, best]
    return np.maximum(W, W.T)


def build_term_graph(d, term, k=K, vote_ids=None, min_votes=MIN_VOTES):
    """Convenience: records -> labels -> matrix -> agreement -> graph for one term."""
    mv = term_records(d, term, vote_ids=vote_ids, min_votes=min_votes)
    lab = member_labels(mv)
    X, vids = vote_matrix(mv, lab.index.to_numpy())
    agree, shared = agreement(X)
    W = knn_graph(agree, k=k)
    return dict(term=term, mv=mv, labels=lab, X=X, vote_ids=vids, agree=agree,
                shared=shared, W=W)
