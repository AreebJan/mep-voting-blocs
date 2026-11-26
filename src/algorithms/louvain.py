"""Louvain community detection, implemented from scratch.

Written by Jazmin (Louvain_Attempt_Jazmin.ipynb). Moved here unchanged except for:
- a resolution parameter `gamma` (gamma=1 is the original algorithm), needed for the
  resolution sweep;
- an optional `verbose` flag instead of always printing each level.

Validated against networkx.community.louvain_communities in the main notebook.

How it works, in two phases that repeat:
1. Local moving: start with every node alone. Visit nodes in random order and move each
   one to the neighbouring community that gives the largest modularity gain, until no
   node moves.
2. Aggregation: merge each community into one "super-node" (H.T @ A @ H sums the edge
   weights between communities; weights inside a community become a self-loop), then
   run phase 1 again on the smaller graph.
Stop when phase 1 no longer merges anything.
"""
import numpy as np
from collections import defaultdict


def modularity(A, labels, gamma=1.0):
    """Q = sum over communities c of [ in_c / 2m  -  gamma * (tot_c / 2m)^2 ]

    in_c  = total weight of edges inside c (each edge counted twice, as in A)
    tot_c = total weighted degree of the nodes in c
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
    """Phase 1. Returns a community label for every node of A."""
    n = A.shape[0]
    labels = np.arange(n, dtype=int)          # each node starts in its own community
    degrees = A.sum(axis=1)
    two_m = degrees.sum()
    if two_m == 0:
        return labels

    totals = {i: float(degrees[i]) for i in range(n)}   # total degree per community
    counts = {i: 1 for i in range(n)}                    # members per community
    next_label = n

    for _ in range(max_passes):
        moves = 0
        for node in rng.permutation(n):                  # random visiting order
            ki = degrees[node]
            if ki == 0:
                continue
            old = int(labels[node])

            # Temporarily remove node from its community
            totals[old] -= ki
            counts[old] -= 1

            # Edge weight from node to each neighbouring community
            links = defaultdict(float)
            for neighbor in np.flatnonzero(A[node]):
                if neighbor != node:
                    links[int(labels[neighbor])] += float(A[node, neighbor])

            # Gain of putting node into community c, up to a constant factor 1/m:
            #   k_i,in(c) - gamma * k_i * tot_c / 2m
            def score(c):
                return links[c] - gamma * ki * totals[c] / two_m

            best, best_score = old, score(old)
            for c in links:
                s = score(c)
                if s > best_score + tol:
                    best, best_score = c, s

            # An empty community scores 0. If every option is negative, start a new one.
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
    """Full Louvain. W: symmetric, non-negative weighted adjacency matrix (numpy).
    Returns a community label (0, 1, 2, ...) for every node of W."""
    A = np.asarray(W, dtype=float).copy()
    assert A.ndim == 2 and A.shape[0] == A.shape[1]
    assert np.allclose(A, A.T)
    assert np.all(A >= 0)
    np.fill_diagonal(A, 0)

    original_A = A.copy()
    original_labels = np.arange(len(A))        # community of each original node
    rng = np.random.default_rng(seed)

    for level in range(max_levels):
        labels = local_moving(A, rng, gamma=gamma)              # phase 1
        _, labels = np.unique(labels, return_inverse=True)      # renumber 0..k-1
        original_labels = labels[original_labels]
        n_communities = labels.max() + 1
        if verbose:
            Q = modularity(original_A, original_labels, gamma)
            print(f"Level {level + 1}: {n_communities} communities, modularity = {Q:.4f}")
        if n_communities == len(A):                             # nothing merged
            break
        H = np.eye(n_communities)[labels]                       # node -> community one-hot
        A = H.T @ A @ H                                         # phase 2: aggregate
    return original_labels
