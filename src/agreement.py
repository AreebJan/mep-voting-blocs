"""MEP x vote matrices, pairwise agreement and the kNN agreement graph.
Uses sparse matrix products, so a whole term takes seconds.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from config import TERMS, POSITIONS, MIN_VOTES, MIN_SHARED, K


def term_records(d, term, vote_ids=None, min_votes=MIN_VOTES):
    """FOR/AGAINST records on contested votes of one term, for MEPs with >= min_votes.
    vote_ids can restrict it to some votes, e.g. one policy area."""
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
    """Group, country and party of each MEP"""
    def mode(s):
        return s.astype("string").fillna("none").mode().iloc[0]
    lab = mv.groupby("member_id", observed=True).agg(
        group=("group_code", mode), country=("country_code", mode), party=("party", mode),
        n_votes=("vote_id", "size"))
    return lab


def vote_matrix(mv, members, vote_ids=None):
    """Sparse MEP x vote matrix: +1 FOR, -1 AGAINST, 0 missing."""
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
    """Share of shared votes where two MEPs voted the same way.
    Since votes are +1/-1, agreement = (shared + X X^T) / (2 * shared).
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
    """Each MEP gets an edge to the k MEPs they agree with most (weight = agreement).
    The graph is made symmetric, and the stable sort makes ties break the same way
    on every machine."""
    n = len(agree)
    a = np.nan_to_num(agree, nan=-1.0)
    W = np.zeros((n, n))
    for i in range(n):
        order = np.argsort(-a[i], kind="stable")
        best = [j for j in order[:k + 1] if j != i and a[i, j] >= 0][:k]
        W[i, best] = agree[i, best]
    return np.maximum(W, W.T)


def build_term_graph(d, term, k=K, vote_ids=None, min_votes=MIN_VOTES):
    """Builds everything above for one term."""
    mv = term_records(d, term, vote_ids=vote_ids, min_votes=min_votes)
    lab = member_labels(mv)
    X, vids = vote_matrix(mv, lab.index.to_numpy())
    agree, shared = agreement(X)
    W = knn_graph(agree, k=k)
    return dict(term=term, mv=mv, labels=lab, X=X, vote_ids=vids, agree=agree,
                shared=shared, W=W)
