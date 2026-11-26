"""A-priori and PCY, our own implementations (course topic: frequent itemsets).

Both take a list of baskets (sets of item names) and return
(frequent, stats): frequent = {frozenset(itemset): count} for all frequent itemsets of
size 1..max_size; stats = candidates counted, frequent itemsets and memory per pass.
They are checked against brute_force in tests/test_frequent_items.py:
    python tests/test_frequent_items.py

Implementation notes
- Items are mapped to integer ids (sorted by name, so the ids and the PCY hash are the
  same on every machine).
- Counting a pass = one pass over the baskets. Baskets are processed in chunks as 0/1
  matrices so numpy does the inner loop; the logic is the same as a basket-by-basket loop.
- Building baskets from the votes, lift and the analyses live in
  analysis/itemset_mining.py.
"""
from itertools import combinations
from collections import Counter
import time
import numpy as np


# --- Reference implementation (for TESTS on small data only) ---------------------------
# --- Reference implementation (for TESTS on small data only) ---------------------------
def brute_force(baskets, min_support, max_size=3):
    """Counts every itemset up to max_size in every basket. Correct but exponential:
    only use it on small toy data to check apriori/pcy.
    min_support: fraction of baskets (0-1) or absolute count (>= 1)."""
    threshold = _threshold(min_support, len(baskets))
    counts = Counter()
    for b in baskets:
        b = sorted(b)
        for k in range(1, max_size + 1):
            counts.update(frozenset(c) for c in combinations(b, k))
    return {s: c for s, c in counts.items() if c >= threshold}


def _threshold(min_support, n_baskets):
    return min_support * n_baskets if min_support < 1 else min_support


# --- Shared building blocks ------------------------------------------------------------
# --- Shared building blocks ------------------------------------------------------------
def _encode(baskets):
    """Item names -> integer ids 0..n-1 (sorted by name), baskets -> arrays of ids."""
    items = sorted(set().union(*baskets))
    index = {name: i for i, name in enumerate(items)}
    encoded = [np.array(sorted(index[x] for x in b), dtype=np.int64) for b in baskets]
    return items, encoded


def _count_items(encoded, n_items):
    """Pass 1 of both algorithms: one counter per item."""
    if not encoded:
        return np.zeros(n_items, dtype=np.int64)
    return np.bincount(np.concatenate(encoded), minlength=n_items)


def _count_candidates(encoded, candidates, n_items, chunk=256):
    """One pass over the baskets, counting how many baskets contain each candidate.

    candidates: int array (n_candidates x k), each row an itemset of item ids.
    For a chunk of baskets we build a 0/1 matrix M (baskets x items); a basket contains
    a candidate if M is 1 in all k of the candidate's columns."""
    counts = np.zeros(len(candidates), dtype=np.int64)
    if len(candidates) == 0:
        return counts
    for start in range(0, len(encoded), chunk):
        part = encoded[start:start + chunk]
        M = np.zeros((len(part), n_items), dtype=bool)
        for r, ids in enumerate(part):
            M[r, ids] = True
        contains = M[:, candidates[:, 0]]
        for col in range(1, candidates.shape[1]):
            contains &= M[:, candidates[:, col]]
        counts += contains.sum(axis=0)
    return counts


def _next_candidates(frequent_prev):
    """A-priori candidate generation for size k from the frequent itemsets of size k-1.

    Join: two frequent (k-1)-sets (as sorted tuples) that share their first k-2 items
          give a k-set (e.g. (a,b) + (a,c) -> (a,b,c)).
    Prune: keep it only if ALL of its (k-1)-subsets are frequent (monotonicity:
          a set cannot be frequent if one of its subsets is not)."""
    prev = sorted(frequent_prev)
    prev_set = set(prev)
    by_prefix = {}
    for t in prev:
        by_prefix.setdefault(t[:-1], []).append(t[-1])
    out = []
    for prefix, lasts in by_prefix.items():
        for a, b in combinations(lasts, 2):          # lasts are sorted, so a < b
            cand = prefix + (a, b)
            if all(sub in prev_set for sub in combinations(cand, len(cand) - 1)):
                out.append(cand)
    return out


def _levels_from(frequent_k, k, encoded, n_items, threshold, max_size, stats, result):
    """A-priori passes k+1, k+2, ... starting from the frequent itemsets of size k."""
    while k < max_size and frequent_k:
        k += 1
        cands = _next_candidates(frequent_k)
        stats["candidates"][k] = len(cands)
        stats["memory_bytes"][k] = 4 * len(cands) * (k + 1)   # itemset + count, 4 B each
        if not cands:
            break
        arr = np.array(cands, dtype=np.int64)
        counts = _count_candidates(encoded, arr, n_items)
        frequent_k = []
        for cand, c in zip(cands, counts):
            if c >= threshold:
                frequent_k.append(cand)
                result[cand] = int(c)
        stats["frequent"][k] = len(frequent_k)
        stats["passes"] += 1


def _decode(result, items):
    return {frozenset(items[i] for i in ids): c for ids, c in result.items()}


# --- A-priori ---------------------------------------------------------------------------
# --- A-priori ---------------------------------------------------------------------------
def apriori(baskets, min_support, max_size=3):
    """A-priori.

    Input:  baskets (list of sets), min_support (fraction 0-1 or absolute count),
            max_size (largest itemset size to find).
    Return: (frequent, stats)
            frequent: dict {frozenset(itemset): count} for all frequent itemsets of size
                      1..max_size (same format as brute_force).
            stats:    candidates counted, frequent itemsets found and memory per pass.

    Pass 1: count single items, keep the frequent ones.
    Pass 2: candidates = every pair of frequent items (stored like the triangular
            matrix of the course: one 4-byte counter per pair).
    Pass k: candidates = k-sets whose (k-1)-subsets are all frequent.
    """
    t0 = time.perf_counter()
    threshold = _threshold(min_support, len(baskets))
    items, encoded = _encode(baskets)
    n_items = len(items)
    stats = dict(candidates={}, frequent={}, memory_bytes={}, passes=1)

    # Pass 1
    item_counts = _count_items(encoded, n_items)
    frequent_items = [i for i in range(n_items) if item_counts[i] >= threshold]
    result = {(i,): int(item_counts[i]) for i in frequent_items}
    stats["candidates"][1] = n_items
    stats["frequent"][1] = len(frequent_items)
    stats["memory_bytes"][1] = 4 * n_items

    # Pass 2: all pairs of frequent items (triangular matrix of m*(m-1)/2 counters)
    if max_size >= 2:
        cands = list(combinations(frequent_items, 2))
        stats["candidates"][2] = len(cands)
        stats["memory_bytes"][2] = 4 * len(cands)
        counts = _count_candidates(encoded, np.array(cands, dtype=np.int64).reshape(-1, 2),
                                   n_items)
        frequent_pairs = [c for c, n in zip(cands, counts) if n >= threshold]
        result.update({c: int(n) for c, n in zip(cands, counts) if n >= threshold})
        stats["frequent"][2] = len(frequent_pairs)
        stats["passes"] += 1
        # Passes 3..max_size
        _levels_from(frequent_pairs, 2, encoded, n_items, threshold, max_size, stats, result)

    stats["time_s"] = time.perf_counter() - t0
    return _decode(result, items), stats


# --- PCY --------------------------------------------------------------------------------
# --- PCY --------------------------------------------------------------------------------
def _hash_pairs(i, j, n_buckets):
    """Deterministic hash of item-id pairs (i < j) to a bucket. Large odd multipliers
    spread consecutive ids over the buckets; ids are fixed by the sorted item names."""
    return (i * 1_000_003 + j * 999_983) % n_buckets


def pcy(baskets, min_support, n_buckets=100_003, max_size=2):
    """PCY (Park-Chen-Yu). Same return format as apriori: (frequent, stats).

    Pass 1: count single items AND hash every pair of every basket into one of n_buckets
            buckets, adding 1 to that bucket's counter. This uses memory A-priori leaves
            idle in pass 1.
    Between passes: replace the bucket counters by a bitmap, 1 bit per bucket
            (1 = bucket count >= threshold). A pair in a bucket with bit 0 cannot be
            frequent, because the whole bucket's count is below the threshold.
    Pass 2: candidate pairs = both items frequent AND the pair hashes to a frequent
            bucket. Candidates are stored with their count (12 bytes per pair in the
            course's "triples" representation), not as a full triangular matrix.
    Pass 3+: A-priori passes from the frequent pairs.
    """
    t0 = time.perf_counter()
    threshold = _threshold(min_support, len(baskets))
    items, encoded = _encode(baskets)
    n_items = len(items)
    stats = dict(candidates={}, frequent={}, memory_bytes={}, passes=1,
                 n_buckets=n_buckets)

    # Pass 1: item counts + bucket counts
    item_counts = _count_items(encoded, n_items)
    buckets = np.zeros(n_buckets, dtype=np.int64)
    for ids in encoded:
        if len(ids) < 2:
            continue
        a, b = np.triu_indices(len(ids), k=1)            # every pair in the basket
        np.add.at(buckets, _hash_pairs(ids[a], ids[b], n_buckets), 1)
    frequent_items = [i for i in range(n_items) if item_counts[i] >= threshold]
    result = {(i,): int(item_counts[i]) for i in frequent_items}
    stats["candidates"][1] = n_items
    stats["frequent"][1] = len(frequent_items)
    stats["memory_bytes"][1] = 4 * n_items + 4 * n_buckets
    stats["bucket_counts_bytes"] = 4 * n_buckets

    if max_size >= 2:
        # Between passes: bitmap of frequent buckets
        bitmap = buckets >= threshold
        stats["frequent_bucket_share"] = float(bitmap.mean())
        stats["bitmap_bytes"] = int(np.ceil(n_buckets / 8))

        # Pass 2: candidate pairs must pass BOTH tests
        all_pairs = np.array(list(combinations(frequent_items, 2)), dtype=np.int64).reshape(-1, 2)
        keep = bitmap[_hash_pairs(all_pairs[:, 0], all_pairs[:, 1], n_buckets)]
        cands = all_pairs[keep]
        stats["pairs_of_frequent_items"] = len(all_pairs)
        stats["candidates"][2] = len(cands)
        stats["memory_bytes"][2] = stats["bitmap_bytes"] + 12 * len(cands)
        counts = _count_candidates(encoded, cands, n_items)
        frequent_pairs = []
        for (i, j), n in zip(cands, counts):
            if n >= threshold:
                frequent_pairs.append((int(i), int(j)))
                result[(int(i), int(j))] = int(n)
        stats["frequent"][2] = len(frequent_pairs)
        stats["passes"] += 1
        _levels_from(frequent_pairs, 2, encoded, n_items, threshold, max_size, stats, result)

    stats["time_s"] = time.perf_counter() - t0
    return _decode(result, items), stats
