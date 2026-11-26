"""The SVD political map (outside technique) and what we measure on it: separation by
group / country / party, the horseshoe (arch) check, whether the far left and far right
vote together, and how one-dimensional voting is per half-year. The SVD itself is in
algorithms/svd.py."""
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from algorithms.svd import ideal_points


def separation(coords, lab, n_perm=100, seed=0, exclude_ni=True):
    """Silhouette of the 2D map by group / country / national party, with a shuffled-label
    baseline. Silhouette: +1 = tight separate clusters, 0 = overlapping, <0 = mixed up."""
    mask = (lab.group != "NI").to_numpy() if exclude_ni else np.ones(len(lab), bool)
    rng = np.random.default_rng(seed)
    rows = []
    for col in ["group", "country", "party"]:
        y = lab[col].to_numpy()[mask]
        # silhouette needs >= 2 members per label; singleton parties are dropped
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
    """Is dimension 2 just a curve of dimension 1 (the 'arch' or 'horseshoe' artefact
    that SVD/PCA produce when the data really lie along ONE gradient)? Returns the R^2 of
    dim 2 explained by a straight line and by a parabola in dim 1. A high quadratic R^2
    means the bend in the map is mostly the artefact, not a real second dimension."""
    x, y = coords[:, 0], coords[:, 1]
    out = {}
    for degree, name in [(1, "R2_linear"), (2, "R2_quadratic")]:
        A = np.vstack([x ** k for k in range(degree + 1)]).T
        resid = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
        out[name] = 1 - resid.var() / y.var()
    return out


def extremes_co_opposition(d, term):
    """Do the far left (The Left) and the far right (ID / PfE) vote together against the
    centre? On votes where EPP, S&D and Renew majorities are on the same side, compare
    how often BOTH extremes oppose them with what independence would predict."""
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
    """Share of voting variance on the first SVD dimension per half-year. Every window
    uses the same number of randomly drawn contested votes (n_votes, repeated reps
    times), so windows with more votes are not favoured."""
    from config import POSITIONS
    from analysis.agreement_graph import vote_matrix
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
