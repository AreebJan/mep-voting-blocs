"""Checks for our A-priori and PCY. Run from the repo root:
    python tests/test_frequent_items.py        (or: pytest tests/)

The checks compare with brute_force() on small baskets where brute force is feasible.
"""
import sys
import random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from algorithms.frequent_itemsets import apriori, pcy, brute_force

# Small textbook-style example: 8 baskets of items a-f
TOY = [set("abc"), set("abd"), set("acd"), set("abcd"), set("be"), set("bce"),
       set("acde"), set("abf")]


def random_baskets(n=300, n_items=25, p=0.3, seed=1):
    rng = random.Random(seed)
    items = [f"P{i:02d}" for i in range(n_items)]
    # items with a lower index are more popular, so there are frequent pairs/triples
    return [{it for j, it in enumerate(items) if rng.random() < p * (1.5 - j / n_items)}
            for _ in range(n)]


def _frequent(result):
    return result[0] if isinstance(result, tuple) else result


def test_apriori_toy():
    for s in (2, 3, 4):
        assert _frequent(apriori(TOY, s, max_size=3)) == brute_force(TOY, s, max_size=3), s


def test_apriori_random():
    b = random_baskets()
    for s in (0.1, 0.2, 0.3):
        assert _frequent(apriori(b, s, max_size=3)) == brute_force(b, s, max_size=3), s


def test_apriori_candidates_reported():
    _, stats = apriori(random_baskets(), 0.2, max_size=3)
    assert "candidates" in stats and set(stats["candidates"]) >= {1, 2}


def test_pcy_pairs_match_apriori():
    b = random_baskets()
    for s in (0.1, 0.2):
        expected = {k: v for k, v in brute_force(b, s, max_size=2).items()}
        assert _frequent(pcy(b, s, n_buckets=101, max_size=2)) == expected, s


def test_pcy_few_buckets_still_correct():
    # with very few buckets almost every bucket is frequent: PCY must still be correct
    b = random_baskets()
    assert _frequent(pcy(b, 0.2, n_buckets=3, max_size=2)) == brute_force(b, 0.2, max_size=2)


def test_pcy_never_more_candidate_pairs_than_apriori():
    b = random_baskets()
    _, sa = apriori(b, 0.2, max_size=2)
    _, sp = pcy(b, 0.2, n_buckets=101, max_size=2)
    assert sp["candidates"][2] <= sa["candidates"][2]


def test_pcy_triples_match_brute_force():
    b = random_baskets(seed=7)
    assert _frequent(pcy(b, 0.1, n_buckets=101, max_size=3)) == brute_force(b, 0.1, max_size=3)


def test_nothing_frequent():
    # threshold above every count: no itemsets, no crash
    assert _frequent(apriori(TOY, 100, max_size=3)) == {}
    assert _frequent(pcy(TOY, 100, n_buckets=11, max_size=3)) == {}


def test_pcy_is_deterministic():
    b = random_baskets()
    assert _frequent(pcy(b, 0.2, n_buckets=101)) == _frequent(pcy(b, 0.2, n_buckets=101))


if __name__ == "__main__":
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_")]
    failed = 0
    for name, f in tests:
        try:
            f()
            print(f"PASS  {name}")
        except NotImplementedError as e:
            failed += 1
            print(f"TODO  {name}: {e}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {name}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
