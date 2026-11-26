"""Checks for our Louvain. Run from the repo root:
    python tests/test_louvain.py        (or: pytest tests/)

networkx is used here only as a reference, never in the analysis.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import numpy as np
import networkx as nx
from sklearn.metrics import normalized_mutual_info_score as nmi
from algorithms.louvain import louvain_from_scratch, modularity


def nx_labels(G, seed=0, gamma=1.0):
    comms = nx.community.louvain_communities(G, weight="weight", seed=seed, resolution=gamma)
    lab = np.empty(G.number_of_nodes(), dtype=int)
    for c, members in enumerate(comms):
        lab[list(members)] = c
    return lab


def planted(n_groups=4, size=25, p_in=0.4, p_out=0.02, seed=1):
    G = nx.planted_partition_graph(n_groups, size, p_in, p_out, seed=seed)
    truth = np.repeat(np.arange(n_groups), size)
    return G, nx.to_numpy_array(G), truth


def test_modularity_matches_networkx():
    G = nx.karate_club_graph()
    W = nx.to_numpy_array(G, weight="weight")
    labels = np.array([0 if G.nodes[i]["club"] == "Mr. Hi" else 1 for i in G])
    expected = nx.community.modularity(G, [set(np.where(labels == c)[0]) for c in (0, 1)],
                                       weight="weight")
    assert abs(modularity(W, labels) - expected) < 1e-9


def test_finds_planted_communities():
    G, W, truth = planted()
    found = louvain_from_scratch(W, seed=0)
    assert nmi(truth, found) > 0.95


def test_modularity_close_to_networkx():
    for seed in range(3):
        G, W, _ = planted(seed=seed + 10, p_out=0.05)
        ours = modularity(W, louvain_from_scratch(W, seed=seed))
        ref = modularity(W, nx_labels(G, seed=seed))
        assert ours >= ref - 0.01, (seed, ours, ref)


def test_weighted_graph():
    G = nx.karate_club_graph()          # has integer edge weights
    W = nx.to_numpy_array(G, weight="weight")
    ours = modularity(W, louvain_from_scratch(W, seed=0))
    ref = modularity(W, nx_labels(G, seed=0))
    assert ours >= ref - 0.02, (ours, ref)


def test_higher_resolution_gives_more_communities():
    G, W, _ = planted(n_groups=6, size=20)
    k_low = louvain_from_scratch(W, seed=0, gamma=0.3).max() + 1
    k_high = louvain_from_scratch(W, seed=0, gamma=3.0).max() + 1
    assert k_high > k_low


def test_isolated_node_stays_alone():
    G, W, _ = planted()
    W = np.pad(W, ((0, 1), (0, 1)))     # add a node with no edges
    labels = louvain_from_scratch(W, seed=0)
    assert np.sum(labels == labels[-1]) == 1


def test_deterministic_for_a_seed():
    G, W, _ = planted()
    assert np.array_equal(louvain_from_scratch(W, seed=3), louvain_from_scratch(W, seed=3))


if __name__ == "__main__":
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_")]
    failed = 0
    for name, f in tests:
        try:
            f()
            print(f"PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {name}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
