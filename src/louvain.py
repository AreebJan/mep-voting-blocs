"""Our own Louvain implementation.


We checked it against networkx's louvain_communities in analysis.ipynb.

The algorithm repeats two steps:
1. Local moving: every node starts in its own community. We go through the nodes in a
   random order and move each one to the neighbouring community where modularity goes
   up the most. Keep going until nothing moves anymore.
2. Aggregation: each community becomes one big node. H.T @ A @ H adds up the edge
   weights between communities (edges inside a community end up on the diagonal).
   Then we do step 1 again on this smaller graph.
We stop when step 1 doesn't merge anything.
"""
import numpy as np
from collections import defaultdict


def modularity(A, labels, gamma=1.0):
    """Modularity of a partition:
    Q = sum over communities c of  in_c / 2m - gamma * (tot_c / 2m)^2

    in_c is the edge weight inside c (counted twice, since A is symmetric)
    tot_c is the sum of the degrees of the nodes in c
    """
    labels = np.asarray(labels)
    degrees = A.sum(axis=1)
    two_m = degrees.sum()
    if two_m == 0:
        return 0.0
    Q = 0.0
    for community in np.unique(labels):
        nodes = np.where(labels == community)[0]
        internal = A[np.ix_(nodes, nodes)].sum()
        total_degree = degrees[nodes].sum()
        Q += internal / two_m
        Q -= gamma * (total_degree / two_m) ** 2
    return Q


def local_moving(A, rng, gamma=1.0, max_passes=30, tol=1e-10):
    """Step 1. Gives back a community label for each node in A."""
    n = A.shape[0]
    labels = np.arange(n, dtype=int)
    degrees = A.sum(axis=1)
    two_m = degrees.sum()
    if two_m == 0:
        return labels

    totals = {i: float(degrees[i]) for i in range(n)}   # sum of degrees in each community
    counts = {i: 1 for i in range(n)}                    # how many nodes in each community
    next_label = n

    for _ in range(max_passes):
        moves = 0
        for node in rng.permutation(n):
            ki = degrees[node]
            if ki == 0:
                continue
            old = int(labels[node])

            # take the node out of its community for now
            totals[old] -= ki
            counts[old] -= 1

            # how much edge weight the node has to each neighbouring community
            links = defaultdict(float)
            for neighbor in np.flatnonzero(A[node]):
                if neighbor != node:
                    links[int(labels[neighbor])] += float(A[node, neighbor])

            # modularity gain of moving the node into c (we drop the 1/m factor
            # since it's the same for every c):  k_i,in(c) - gamma * k_i * tot_c / 2m
            def score(c):
                return links[c] - gamma * ki * totals[c] / two_m

            best, best_score = old, score(old)
            for c in links:
                s = score(c)
                if s > best_score + tol:
                    best, best_score = c, s

            # being alone gives a score of 0, so if all the options are negative
            # the node gets a new community of its own
            if counts[old] > 0 and best_score < -tol:
                best = next_label
                next_label += 1
                totals[best] = 0.0
                counts[best] = 0

            labels[node] = best
            totals[best] += ki
            counts[best] += 1
            if best != old:
                moves += 1
        if moves == 0:
            break
    return labels


def louvain_from_scratch(W, seed=42, gamma=1.0, max_levels=20, verbose=False):
    """Runs Louvain on a symmetric, non-negative weight matrix W.
    Gives back a community number for each node."""
    A = np.asarray(W, dtype=float).copy()
    assert A.ndim == 2 and A.shape[0] == A.shape[1]
    assert np.allclose(A, A.T)
    assert np.all(A >= 0)
    np.fill_diagonal(A, 0)

    original_A = A.copy()
    original_labels = np.arange(len(A))        # which community each original node is in
    rng = np.random.default_rng(seed)

    for level in range(max_levels):
        labels = local_moving(A, rng, gamma=gamma)
        _, labels = np.unique(labels, return_inverse=True)      # relabel as 0..k-1
        original_labels = labels[original_labels]
        n_communities = labels.max() + 1
        if verbose:
            Q = modularity(original_A, original_labels, gamma)
            print(f"Level {level + 1}: {n_communities} communities, modularity = {Q:.4f}")
        if n_communities == len(A):                             # nothing merged
            break
        H = np.eye(n_communities)[labels]
        A = H.T @ A @ H                                         # step 2: merge communities
    return original_labels
