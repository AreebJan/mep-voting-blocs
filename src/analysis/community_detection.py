"""Community detection on the agreement graph (course topic: graph clustering).

Validation of our Louvain against networkx, the main comparison of communities with
groups / countries / national parties, how groups split, the resolution sweep, policy
areas and robustness. Used for RQ1."""
import time
import numpy as np
import pandas as pd
import networkx as nx
from config import SEEDS, GAMMA, N_PERMUTATIONS, MIN_VOTES_POLICY
from algorithms.louvain import louvain_from_scratch, modularity
from analysis.evaluation import compare, label_modularity, nx_partition, nmi, ami
from analysis.agreement_graph import build_term_graph, knn_graph

LABELS = ["group", "country", "party"]


# --- Validation of our Louvain ---------------------------------------------------------
def validate_louvain(W, seeds=SEEDS):
    """Our Louvain vs networkx on the same graph and the same seeds."""
    G = nx.from_numpy_array(W)
    rows, ours, ref = [], [], []
    for s in seeds:
        t = time.perf_counter(); a = louvain_from_scratch(W, seed=s); ta = time.perf_counter() - t
        t = time.perf_counter(); b = nx_partition(G, s); tb = time.perf_counter() - t
        ours.append(a); ref.append(b)
        rows.append(dict(seed=s, Q_ours=modularity(W, a), Q_networkx=modularity(W, b),
                         k_ours=a.max() + 1, k_networkx=b.max() + 1,
                         time_ours_s=ta, time_networkx_s=tb))
    per_seed = pd.DataFrame(rows)
    pairs = lambda P: [nmi(P[i], P[j]) for i in range(len(P)) for j in range(i + 1, len(P))]
    summary = dict(
        Q_ours=per_seed.Q_ours.mean(), Q_ours_sd=per_seed.Q_ours.std(),
        Q_networkx=per_seed.Q_networkx.mean(), Q_networkx_sd=per_seed.Q_networkx.std(),
        NMI_ours_vs_networkx=np.mean([nmi(a, b) for a in ours for b in ref]),
        NMI_ours_vs_ours=np.mean(pairs(ours)), NMI_networkx_vs_networkx=np.mean(pairs(ref)),
        time_ours_s=per_seed.time_ours_s.mean(), time_networkx_s=per_seed.time_networkx_s.mean())
    return per_seed, summary, ours


def best_of_seeds(W, seeds=SEEDS, gamma=GAMMA):
    """Run Louvain for every seed; return the highest-modularity partition and all runs."""
    runs = [louvain_from_scratch(W, seed=s, gamma=gamma) for s in seeds]
    Q = [modularity(W, r, gamma) for r in runs]
    return runs[int(np.argmax(Q))], runs, Q


# --- Main comparison -------------------------------------------------------------------
def main_table(g, comm, runs=None, n_perm=N_PERMUTATIONS):
    """One row per label set: NMI/AMI/ARI (+ chance), modularity of the label partition.
    If runs is given, also the spread of AMI over seeds."""
    W, lab = g["W"], g["labels"]
    rows = []
    for col in LABELS:
        r = compare(lab[col], comm, n_perm=n_perm)
        r.update(label=col, n_labels=lab[col].nunique(),
                 Q_of_label_partition=label_modularity(W, lab[col]))
        if runs is not None:
            vals = [ami(lab[col], x) for x in runs]
            r.update(AMI_min_over_seeds=min(vals), AMI_max_over_seeds=max(vals))
        rows.append(r)
    out = pd.DataFrame(rows).set_index("label")
    out.attrs["Q_louvain"] = modularity(W, comm)
    out.attrs["n_communities"] = int(comm.max() + 1)
    return out


def splits_follow_nationality(lab, comm, min_group=20, exclude=("NI",)):
    """Inside each group, are two MEPs of the same national party more often in the same
    community than two MEPs of different national parties of that group?

    Returns per-group shares and a pooled row. If splits ignored nationality, the two
    shares would be equal."""
    df = lab.assign(c=comm)
    rows = []
    tot = np.zeros(4)
    for grp, d in df.groupby("group"):
        if grp in exclude or len(d) < min_group:
            continue
        i, j = np.triu_indices(len(d), 1)
        same_c = d.c.to_numpy()[i] == d.c.to_numpy()[j]
        same_p = d.party.to_numpy()[i] == d.party.to_numpy()[j]
        n_comm = (d.c.value_counts() >= 5).sum()
        rows.append(dict(group=grp, size=len(d), communities_with_5plus=n_comm,
                         together_same_party=same_c[same_p].mean(),
                         together_different_party=same_c[~same_p].mean(),
                         pairs_same_party=int(same_p.sum())))
        tot += [same_c[same_p].sum(), same_p.sum(), same_c[~same_p].sum(), (~same_p).sum()]
    out = pd.DataFrame(rows)
    pooled = dict(group="ALL (pooled)", size=int(out["size"].sum()),
                  communities_with_5plus=np.nan,
                  together_same_party=tot[0] / tot[1],
                  together_different_party=tot[2] / tot[3], pairs_same_party=int(tot[1]))
    return pd.concat([out, pd.DataFrame([pooled])], ignore_index=True)


# --- Resolution sweep ------------------------------------------------------------------
def resolution_sweep(g, gammas, seeds=SEEDS[:5]):
    W, lab = g["W"], g["labels"]
    rows = []
    for gamma in gammas:
        comm, runs, Q = best_of_seeds(W, seeds=seeds, gamma=gamma)
        sizes = np.bincount(comm)
        row = dict(gamma=gamma, n_communities=len(sizes),
                   n_communities_10plus=int((sizes >= 10).sum()),
                   largest_share=sizes.max() / len(comm))
        for col in LABELS:
            row[f"AMI_{col}"] = ami(lab[col], comm)
            row[f"NMI_{col}"] = nmi(lab[col], comm)
        rows.append(row)
    ref = {f"n_{col}": lab[col].nunique() for col in LABELS}
    out = pd.DataFrame(rows)
    out.attrs.update(ref)
    return out


# --- Policy areas ----------------------------------------------------------------------
def policy_areas(d, term, min_votes=MIN_VOTES_POLICY, seeds=SEEDS[:5]):
    """Rebuild the graph on the contested votes of one top-level OEIL policy area at a
    time. MEPs need min_votes FOR/AGAINST votes in that area."""
    topics = d["vote_topics"]
    votes = d["votes"]
    term_contested = set(votes.vote_id[(votes.term == term) & votes.contested])
    rows = []
    for (topic, label), t in topics.groupby(["topic", "label"]):
        ids = set(t.vote_id) & term_contested
        if len(ids) < 2 * min_votes:
            rows.append(dict(topic=topic, label=label, n_votes=len(ids), skipped=True))
            continue
        g = build_term_graph(d, term, vote_ids=ids, min_votes=min_votes)
        comm, runs, Q = best_of_seeds(g["W"], seeds=seeds)
        lab = g["labels"]
        row = dict(topic=topic, label=label, n_votes=len(ids), n_meps=len(lab),
                   n_communities=int(comm.max() + 1), Q=modularity(g["W"], comm), skipped=False)
        for col in LABELS:
            r = compare(lab[col], comm, n_perm=50)
            row[f"AMI_{col}"] = r["AMI"]
            row[f"NMI_{col}"] = r["NMI"]
            row[f"NMI_{col}_chance"] = r["NMI_chance"]
        rows.append(row)
    return pd.DataFrame(rows)


# --- Robustness ------------------------------------------------------------------------
def robustness(g, ks=(10, 15, 25), seeds=SEEDS[:5]):
    """Same analysis with a different number of neighbours, and with NI excluded.
    Reuses the term's agreement matrix; only the kNN graph is rebuilt."""
    rows = []
    lab = g["labels"]
    for k in ks:
        W = knn_graph(g["agree"], k=k)
        comm, runs, Q = best_of_seeds(W, seeds=seeds)
        for name, mask in [("all MEPs", np.ones(len(lab), bool)),
                           ("NI excluded", (lab.group != "NI").to_numpy())]:
            rows.append(dict(k=k, subset=name, n_meps=int(mask.sum()),
                             n_communities=int(comm.max() + 1), Q=modularity(W, comm),
                             **{f"AMI_{c}": ami(lab[c][mask], comm[mask]) for c in LABELS},
                             **{f"ARI_{c}": compare(lab[c][mask], comm[mask], n_perm=1)["ARI"]
                                for c in LABELS}))
    return pd.DataFrame(rows)
