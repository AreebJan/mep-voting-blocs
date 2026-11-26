"""RQ2: which cross-border coalitions of national parties pass votes?

One basket per contested vote; the items are the national parties (or groups) whose
majority voted FOR. A frequent itemset {A, B, C} = parties that very often vote FOR
together.

`apriori` and `pcy` are our own implementations (course algorithms). They are checked
against a brute-force count in tests/test_frequent_items.py:
    python tests/test_frequent_items.py

Why A-priori is needed here (useful for the report): a term-9 party basket has ~80
items on average, so one basket alone contains ~3,200 pairs and ~82,000 triples;
counting every triple of every basket directly is ~7 * 10^8 increments. A-priori only
counts the itemsets whose subsets are all frequent. Run `basket_stats` for exact numbers.

Implementation notes
- Items are mapped to integer ids (sorted by name, so the ids and the PCY hash are the
  same on every machine).
- Counting a pass = one pass over the baskets. Baskets are processed in chunks as 0/1
  matrices so numpy does the inner loop; the logic is the same as a basket-by-basket loop.
"""
from itertools import combinations
from collections import Counter
import time
import numpy as np
import pandas as pd
from config import POSITIONS, MIN_PARTY_VOTERS


# --- Baskets (ready) -------------------------------------------------------------------
def build_baskets(d, term, level="party", passed_only=False,
                  min_voters=MIN_PARTY_VOTERS, vote_ids=None):
    """Returns (baskets, vote_ids): baskets is a list of frozensets of item names.

    level="party": items are national parties as "COUNTRY:label" (stable across terms)
    level="group": items are European political groups (NI excluded, not a group)
    passed_only:   keep only votes that were adopted ("coalitions that pass votes")
    min_voters:    an item needs at least this many FOR/AGAINST voters on the vote
    """
    v = d["votes"]
    v = v[(v.term == term) & v.contested]
    if passed_only:
        v = v[v.passed]
    if vote_ids is not None:
        v = v[v.vote_id.isin(vote_ids)]
    mv = d["member_votes"]
    mv = mv[mv.vote_id.isin(v.vote_id) & mv.position.isin(POSITIONS)]
    col = "party" if level == "party" else "group_code"
    x = pd.DataFrame({"vote_id": mv.vote_id.to_numpy(),
                      "item": mv[col].astype("string").to_numpy(),
                      "for_": (mv.position == "FOR").to_numpy()})
    if level == "group":
        x = x[x.item != "NI"]
    else:
        x = x[~x.item.str.endswith(":none")]          # MEPs without a national party
    agg = x.groupby(["vote_id", "item"]).for_.agg(["sum", "size"]).reset_index()
    agg = agg[(agg["size"] >= min_voters) & (agg["sum"] > agg["size"] / 2)]
    by_vote = agg.groupby("vote_id").item.apply(frozenset)
    ids = v.vote_id[v.vote_id.isin(by_vote.index)].to_numpy()
    return [by_vote[i] for i in ids], ids


def basket_stats(baskets):
    sizes = np.array([len(b) for b in baskets])
    items = set().union(*baskets)
    return dict(baskets=len(baskets), distinct_items=len(items),
                mean_basket_size=sizes.mean(),
                pairs_if_counted_naively=int(sum(s * (s - 1) // 2 for s in sizes)),
                triples_if_counted_naively=int(sum(s * (s - 1) * (s - 2) // 6 for s in sizes)))


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


# --- Validation helpers (ready) --------------------------------------------------------
def to_table(frequent, n_baskets, min_size=2):
    """Frequent itemsets as a sorted DataFrame with support and lift.
    lift = support(S) / product of the supports of its items. lift > 1 means the items
    vote FOR together more often than if each voted independently (the baseline)."""
    single = {next(iter(s)): c for s, c in frequent.items() if len(s) == 1}
    rows = []
    for s, c in frequent.items():
        if len(s) < min_size:
            continue
        expected = np.prod([single[i] / n_baskets for i in s])
        rows.append(dict(itemset=" + ".join(sorted(s)), size=len(s), count=c,
                         support=c / n_baskets, lift=(c / n_baskets) / expected))
    cols = ["itemset", "size", "count", "support", "lift"]
    return pd.DataFrame(rows, columns=cols).sort_values("support", ascending=False)


def required_coalitions(d, term=10):
    """Answer key: the minimal combinations of groups whose FOR votes a vote needed to
    pass (HowTheyVote's groups_required_for_majority.csv, available from 2024 only, so
    term 10 only). Returns how often each combination appears, per vote.

    Suggested comparison: build group-level baskets on the same votes
    (build_baskets(d, 10, level="group", passed_only=True, vote_ids=<these ids>)) and
    check that the frequent group itemsets found by apriori/pcy contain the most common
    required combinations, and with what support."""
    req = d["required"]
    v = d["votes"]
    ids = set(v.vote_id[(v.term == term) & v.contested & v.passed])
    req = req[req.vote_id.isin(ids)]
    counts = req.group_codes.value_counts().rename_axis("required_groups").reset_index(name="votes")
    counts["share_of_votes"] = counts.votes / req.vote_id.nunique()
    return counts, np.array(sorted(req.vote_id.unique()))
