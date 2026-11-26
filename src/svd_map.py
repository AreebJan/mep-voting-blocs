"""Out of course topic: ideal-point scaling with the SVD of the MEP x vote matrix.

If MEPs have positions on a few hidden political dimensions, the vote matrix is roughly
low-rank, so the first two singular vectors give each MEP a place on a 2D map.

1. X: +1 FOR, -1 AGAINST, 0 missing (contested votes of one term).
2. Centre each vote on the mean of the MEPs who voted; missing entries stay 0.
3. SVD: Xc = U S V^T. Coordinates = Xc V[:, :2].
4. Divide by the share of votes each MEP attended, otherwise MEPs who missed many votes
   get pulled towards the centre.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score


def ideal_points(X, n_dims=2):
    """Returns the MEP coordinates and the share of variance per dimension."""
    Xd = X.toarray()
    O = Xd != 0
    col_mean = Xd.sum(0) / np.maximum(O.sum(0), 1)
    Xc = np.where(O, Xd - col_mean, 0.0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    explained = S ** 2 / (S ** 2).sum()
    coords = (Xc @ Vt[:n_dims].T) / O.mean(1, keepdims=True)
    return coords, explained


def orient(coords, labels, left="GUE_NGL", up=None):
    """Flips the signs so the `left` group is on the left of dimension 1."""
    c = coords.copy()
    if c[labels == left, 0].mean() > 0:
        c[:, 0] *= -1
    if up is not None and c[labels == up, 1].mean() < 0:
        c[:, 1] *= -1
    return c


def separation(coords, lab, n_perm=100, seed=0, exclude_ni=True):
    """Silhouette of the map by group, country and party, against shuffled labels.
    +1 = clearly separate clusters, 0 = overlapping, below 0 = mixed up."""
    mask = (lab.group != "NI").to_numpy() if exclude_ni else np.ones(len(lab), bool)
    rng = np.random.default_rng(seed)
    rows = []
    for col in ["group", "country", "party"]:
        y = lab[col].to_numpy()[mask]
        # silhouette needs at least 2 members per label
        vc = pd.Series(y).value_counts()
        keep = np.isin(y, vc[vc >= 2].index)
        x, y = coords[mask][keep], y[keep]
        obs = silhouette_score(x, y)
        null = [silhouette_score(x, rng.permutation(y)) for _ in range(n_perm)]
        rows.append(dict(label=col, silhouette=obs, chance_mean=np.mean(null),
                         chance_p95=np.percentile(null, 95), n_meps=len(y),
                         n_labels=len(np.unique(y))))
    return pd.DataFrame(rows)


def group_centres(coords, lab):
    df = pd.DataFrame(coords[:, :2], columns=["dim1", "dim2"]).assign(group=lab.group.to_numpy())
    return df.groupby("group").agg(dim1=("dim1", "mean"), dim2=("dim2", "mean"),
                                   n=("dim1", "size")).sort_values("dim1")


def arch_check(coords):
    """Checks if dimension 2 is just the horseshoe artefact of SVD: R^2 of dim 2 from a
    line and from a parabola in dim 1. A high quadratic R^2 means the bend isn't a real
    second dimension."""
    x, y = coords[:, 0], coords[:, 1]
    out = {}
    for degree, name in [(1, "R2_linear"), (2, "R2_quadratic")]:
        A = np.vstack([x ** k for k in range(degree + 1)]).T
        resid = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
        out[name] = 1 - resid.var() / y.var()
    return out


def extremes_co_opposition(d, term):
    """On votes where EPP, S&D and Renew agree, how often do The Left and the far right
    both vote against them, compared to if they were independent?"""
    from config import POSITIONS
    V, mv = d["votes"], d["member_votes"]
    v = V[(V.term == term) & V.contested]
    x = mv[mv.vote_id.isin(v.vote_id) & mv.position.isin(POSITIONS)]
    x = pd.DataFrame({"vote_id": x.vote_id.to_numpy(), "group": x.group_code.astype(str).to_numpy(),
                      "pos": np.where(x.position.to_numpy() == "FOR", 1, -1)})
    m = np.sign(x.groupby(["vote_id", "group"]).pos.sum().unstack(fill_value=0))
    far = "ID" if term == 9 else "PFE"
    centre = (m.EPP == m.SD) & (m.SD == m.RENEW) & (m.EPP != 0)
    c = m[centre]
    left_opp = (c.GUE_NGL == -c.EPP).mean()
    right_opp = (c[far] == -c.EPP).mean()
    both = ((c.GUE_NGL == -c.EPP) & (c[far] == -c.EPP)).mean()
    agree = ((m.GUE_NGL == m[far]) & (m.GUE_NGL != 0)).sum() / ((m.GUE_NGL != 0) & (m[far] != 0)).sum()
    return dict(term=term, far_right=far, centre_united_share=centre.mean(),
                left_opposes=left_opp, far_right_opposes=right_opp, both_oppose=both,
                both_if_independent=left_opp * right_opp,
                ratio_to_independent=both / (left_opp * right_opp),
                left_far_right_majority_agreement=agree)


def dimensionality_over_time(d, n_votes=500, min_votes=200, reps=5, seed=0):
    """Share of variance on the first SVD dimension per half-year. Each window uses the
    same number of random votes, so windows with more votes aren't favoured."""
    from config import POSITIONS
    from agreement import vote_matrix
    V, mv = d["votes"], d["member_votes"]
    v = V[(V.term > 0) & V.contested][["vote_id", "timestamp", "term"]].copy()
    v["half"] = v.timestamp.dt.year.astype(str) + np.where(v.timestamp.dt.month <= 6, "-H1", "-H2")
    rng = np.random.default_rng(seed)
    rows = []
    for half, w in v.groupby("half"):
        ids = w.vote_id.to_numpy()
        row = dict(half=half, term=int(w.term.mode().iloc[0]), contested_votes=len(ids))
        if len(ids) >= n_votes:
            shares = []
            for _ in range(reps):
                pick = rng.choice(ids, n_votes, replace=False)
                x = mv[mv.vote_id.isin(pick) & mv.position.isin(POSITIONS)]
                n = x.groupby("member_id").size()
                x = x[x.member_id.isin(n[n >= min_votes].index)]
                X, _ = vote_matrix(x, np.sort(x.member_id.unique()))
                _, ex = ideal_points(X)
                shares.append(ex[:2])
            shares = np.array(shares)
            row.update(dim1_share=shares[:, 0].mean(), dim1_sd=shares[:, 0].std(),
                       dim2_share=shares[:, 1].mean())
        rows.append(row)
    return pd.DataFrame(rows)
