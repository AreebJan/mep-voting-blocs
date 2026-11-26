"""
Party or passport? Starter pipeline for the European Parliament voting project.

Builds an MEP-to-MEP agreement graph for one parliamentary term, runs Louvain,
and checks whether communities follow political groups, countries or national parties.

Data: HowTheyVote.eu export (open data, updated weekly)
  https://github.com/HowTheyVote/data/releases/latest/download/export.zip
Unzip it and pass the folder as the first argument:
  python ep_starter.py path/to/export 10
"""
import sys
import numpy as np
import pandas as pd
import scipy.sparse as sp
import networkx as nx
from sklearn.metrics import normalized_mutual_info_score as nmi, adjusted_rand_score as ari

DATA = sys.argv[1] if len(sys.argv) > 1 else "export"
TERM = int(sys.argv[2]) if len(sys.argv) > 2 else 10

# Term 9 starts after Brexit (UK MEPs left 31 Jan 2020) so membership is stable.
TERMS = {9: ("2020-02-01", "2024-07-16"), 10: ("2024-07-16", "2030-01-01")}
LOPSIDED = 0.90   # drop votes where >90% took the same position (little information)
MIN_VOTES = 500   # drop MEPs who took part in fewer contested votes
K = 15            # neighbours kept per MEP in the kNN backbone graph


def load_term(term):
    votes = pd.read_csv(f"{DATA}/votes.csv", parse_dates=["timestamp"])
    lo, hi = TERMS[term]
    ids = set(votes.id[(votes.timestamp >= lo) & (votes.timestamp < hi)])
    mv = pd.read_csv(f"{DATA}/member_votes.csv",
                     usecols=["vote_id", "member_id", "position", "country_code",
                              "group_code", "national_party_short_label"])
    mv = mv[mv.vote_id.isin(ids) & mv.position.isin(["FOR", "AGAINST", "ABSTENTION"])]
    top_share = mv.groupby("vote_id").position.agg(lambda s: s.value_counts(normalize=True).iloc[0])
    contested = top_share[top_share < LOPSIDED].index
    mv = mv[mv.vote_id.isin(contested)]
    n = mv.groupby("member_id").size()
    mv = mv[mv.member_id.isin(n[n >= MIN_VOTES].index)]
    return mv, contested


def agreement_matrix(mv, vote_ids):
    """Share of co-attended votes where two MEPs took the same position (sparse matmul)."""
    labels = mv.groupby("member_id").agg(
        group=("group_code", lambda s: s.mode()[0]),       # group held for most votes
        country=("country_code", "first"),
        party=("national_party_short_label", lambda s: s.mode()[0]))
    mi = {m: i for i, m in enumerate(labels.index)}
    vi = {v: i for i, v in enumerate(vote_ids)}
    r, c = mv.member_id.map(mi).values, mv.vote_id.map(vi).values

    def onehot(pos):
        k = (mv.position == pos).values
        return sp.csr_matrix((np.ones(k.sum()), (r[k], c[k])), shape=(len(mi), len(vi)))

    F, A, B = onehot("FOR"), onehot("AGAINST"), onehot("ABSTENTION")
    present = F + A + B
    same = (F @ F.T + A @ A.T + B @ B.T).toarray()
    both = (present @ present.T).toarray()
    agree = same / np.maximum(both, 1)
    np.fill_diagonal(agree, 0)
    return agree, labels


def knn_graph(agree, k=K):
    G = nx.Graph()
    G.add_nodes_from(range(len(agree)))
    for i in range(len(agree)):
        for j in np.argsort(-agree[i])[:k]:
            G.add_edge(i, int(j), weight=float(agree[i, j]))
    return G


def partition_from_labels(values):
    return [set(np.where(values == x)[0]) for x in pd.unique(values)]


if __name__ == "__main__":
    mv, vote_ids = load_term(TERM)
    agree, labels = agreement_matrix(mv, vote_ids)
    G = knn_graph(agree)
    # TODO (project): replace with your own Louvain implementation and compare.
    comms = nx.community.louvain_communities(G, weight="weight", seed=1)
    part = np.empty(len(labels), int)
    for ci, members in enumerate(comms):
        part[list(members)] = ci
    labels["community"] = part

    print(f"Term {TERM}: {len(vote_ids)} contested votes, {len(labels)} MEPs, {len(comms)} communities")
    for name in ["group", "country", "party"]:
        y = labels[name].values
        Q = nx.community.modularity(G, partition_from_labels(y))
        print(f"  vs {name:8s} NMI={nmi(y, part):.3f}  ARI={ari(y, part):.3f}  modularity of {name} partition={Q:.3f}")
    print(f"  Louvain modularity={nx.community.modularity(G, comms):.3f}")
    print("\nCommunity makeup:")
    for ci, d in labels.groupby("community"):
        print(f"  [{ci}] n={len(d):3d}  groups={d.group.value_counts().head(2).to_dict()}  "
              f"countries={d.country.value_counts().head(3).to_dict()}")
