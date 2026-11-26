"""Comparing partitions with label sets, always next to a chance baseline.

NMI  normalised mutual information: 1 = same partition, but NOT 0 for random partitions
     when there are many small labels (e.g. ~200 national parties), so it needs a baseline.
AMI  adjusted mutual information: NMI corrected for chance, about 0 for random partitions.
ARI  adjusted Rand index: 1 = identical, about 0 = chance level, based on pairs of MEPs.
"""
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics import (normalized_mutual_info_score as nmi,
                             adjusted_mutual_info_score as ami,
                             adjusted_rand_score as ari)
from algorithms.louvain import modularity


def compare(truth, pred, n_perm=200, seed=0):
    """NMI/AMI/ARI of pred vs truth, plus the mean NMI/ARI when pred is shuffled."""
    truth, pred = np.asarray(truth), np.asarray(pred)
    rng = np.random.default_rng(seed)
    null_nmi, null_ari = [], []
    for _ in range(n_perm):
        p = rng.permutation(pred)
        null_nmi.append(nmi(truth, p))
        null_ari.append(ari(truth, p))
    return dict(NMI=nmi(truth, pred), AMI=ami(truth, pred), ARI=ari(truth, pred),
                NMI_chance=np.mean(null_nmi), NMI_chance_p95=np.percentile(null_nmi, 95),
                ARI_chance=np.mean(null_ari))


def label_modularity(W, labels):
    """Modularity of a partition given by labels (e.g. 'everyone in their group')."""
    return modularity(W, pd.factorize(np.asarray(labels))[0])


def nx_partition(G, seed):
    """networkx Louvain as an integer label array (used only for validation)."""
    comms = nx.community.louvain_communities(G, weight="weight", seed=seed)
    lab = np.empty(G.number_of_nodes(), dtype=int)
    for c, members in enumerate(comms):
        lab[list(members)] = c
    return lab


def describe_communities(labels_df, comm, top=3):
    """Table with size and the top groups / countries / parties of each community."""
    df = labels_df.assign(community=comm)
    rows = []
    for c, g in df.groupby("community"):
        def top_of(col):
            vc = g[col].value_counts()
            return ", ".join(f"{k} {v}" for k, v in vc.head(top).items())
        rows.append(dict(community=c, size=len(g),
                         main_group=g.group.value_counts().index[0],
                         main_group_share=round(g.group.value_counts().iloc[0] / len(g), 2),
                         groups=top_of("group"), countries=top_of("country"),
                         parties=top_of("party")))
    return pd.DataFrame(rows).sort_values("size", ascending=False).reset_index(drop=True)
