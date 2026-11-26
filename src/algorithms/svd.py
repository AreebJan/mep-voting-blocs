"""Outside technique: ideal-point scaling with the SVD of the MEP x vote matrix.

Idea (as in NOMINATE-style scaling in political science): if MEPs have positions on a
few hidden political dimensions, their FOR/AGAINST votes are roughly a low-rank matrix.
The SVD finds the best rank-2 approximation, and each MEP's coordinates on the first two
singular vectors place them on a 2D "political map".

Steps
1. X: +1 FOR, -1 AGAINST, 0 missing (contested votes of one term).
2. Centre each vote on the mean of the MEPs who voted (so a vote everyone supports does
   not dominate); missing entries stay 0 = "no information".
3. SVD: Xc = U S V^T. Coordinates = Xc V[:, :2].
4. Missing-data correction: an MEP who attended half the votes would otherwise sit
   halfway to the centre. We divide by the share of votes they attended.
"""
import numpy as np


def ideal_points(X, n_dims=2):
    """X: sparse MEP x vote matrix (+1/-1/0). Returns coordinates, explained share."""
    Xd = X.toarray()
    O = Xd != 0
    col_mean = Xd.sum(0) / np.maximum(O.sum(0), 1)
    Xc = np.where(O, Xd - col_mean, 0.0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    explained = S ** 2 / (S ** 2).sum()
    coords = (Xc @ Vt[:n_dims].T) / O.mean(1, keepdims=True)
    return coords, explained


def orient(coords, labels, left="GUE_NGL", up=None):
    """Flip signs so that dimension 1 has the `left` group on the left (negative)."""
    c = coords.copy()
    if c[labels == left, 0].mean() > 0:
        c[:, 0] *= -1
    if up is not None and c[labels == up, 1].mean() < 0:
        c[:, 1] *= -1
    return c
