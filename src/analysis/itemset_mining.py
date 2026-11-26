"""Frequent-itemset mining on vote baskets (course topic: frequent itemsets).

Two kinds of baskets:
- "FOR" baskets: per adopted contested vote, the national parties (or groups) whose
  majority voted FOR. Dense (~55% of parties in every basket).
- "rebel" baskets: per contested vote, the national parties whose majority voted
  AGAINST the rest of their own European group. Sparse (~6 parties per vote).
Comparing A-priori and PCY on both shows when PCY's hashing pays off. Also: lift tables,
rebel factions, group coalitions in term 9 vs term 10, and a check against HowTheyVote's
groups_required_for_majority. Used for RQ2."""
import numpy as np
import pandas as pd
from config import POSITIONS, MIN_PARTY_VOTERS
from algorithms import frequent_itemsets as fi


# --- Baskets ----------------------------------------------------------------------------
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


# --- Baskets ----------------------------------------------------------------------------
def _records(d, term):
    V, mv = d["votes"], d["member_votes"]
    v = V[(V.term == term) & V.contested]
    x = mv[mv.vote_id.isin(v.vote_id) & mv.position.isin(POSITIONS)]
    return v, pd.DataFrame({"vote_id": x.vote_id.to_numpy(),
                            "group": x.group_code.astype(str).to_numpy(),
                            "party": x.party.astype(str).to_numpy(),
                            "pos": np.where(x.position.to_numpy() == "FOR", 1, -1)})


def rebel_baskets(d, term, min_voters=MIN_PARTY_VOTERS):
    """Per contested vote: national parties whose majority voted against the majority of
    the REST of their European group (the party's own MEPs are left out of the group
    line). NI is not a group and is skipped. Returns (baskets, vote_ids, party->group)."""
    v, x = _records(d, term)
    x = x[(x.group != "NI") & ~x.party.str.endswith(":none")]
    gnet = x.groupby(["vote_id", "group"]).pos.sum().rename("gnet")
    p = x.groupby(["vote_id", "group", "party"]).pos.agg(["sum", "size"]).reset_index()
    p = p.join(gnet, on=["vote_id", "group"])
    p["rest_of_group"] = np.sign(p.gnet - p["sum"])
    p = p[(p["size"] >= min_voters) & (p.rest_of_group != 0)]
    rebels = p[np.sign(p["sum"]) == -p.rest_of_group]
    by_vote = rebels.groupby("vote_id").party.apply(frozenset)
    ids = v.vote_id.to_numpy()
    party_group = p.groupby("party").group.agg(lambda s: s.mode().iloc[0])
    return [by_vote.get(i, frozenset()) for i in ids], ids, party_group


# --- A-priori vs PCY --------------------------------------------------------------------
# --- 1. A-priori vs PCY -----------------------------------------------------------------
def compare_algorithms(baskets_by_name, supports, max_size=3, n_buckets=10_007):
    """Runs both algorithms on every basket set and support; checks they agree."""
    rows = []
    for name, baskets in baskets_by_name.items():
        for s in supports:
            fa, sa = fi.apriori(baskets, s, max_size=max_size)
            fp, sp = fi.pcy(baskets, s, n_buckets=n_buckets, max_size=max_size)
            assert fa == fp, f"A-priori and PCY disagree on {name}, support {s}"
            rows.append(dict(
                baskets=name, support=s, n_baskets=len(baskets),
                mean_basket_size=np.mean([len(b) for b in baskets]),
                frequent_items=sa["frequent"].get(1, 0), frequent_pairs=sa["frequent"].get(2, 0),
                frequent_triples=sa["frequent"].get(3, 0),
                apriori_pair_candidates=sa["candidates"].get(2, 0),
                pcy_pair_candidates=sp["candidates"].get(2, 0),
                pcy_candidate_share=sp["candidates"].get(2, 0) / max(sa["candidates"].get(2, 0), 1),
                frequent_bucket_share=sp.get("frequent_bucket_share", np.nan),
                apriori_pass2_bytes=sa["memory_bytes"].get(2, 0),
                pcy_pass2_bytes=sp["memory_bytes"].get(2, 0),
                apriori_s=sa["time_s"], pcy_s=sp["time_s"]))
    return pd.DataFrame(rows)


def bucket_sweep(baskets, support, bucket_counts=(101, 1_009, 10_007, 100_003)):
    """Effect of the number of PCY buckets on candidate pairs and pass-2 memory."""
    rows = []
    for nb in bucket_counts:
        _, sp = fi.pcy(baskets, support, n_buckets=nb, max_size=2)
        rows.append(dict(n_buckets=nb, frequent_bucket_share=sp["frequent_bucket_share"],
                         pcy_pair_candidates=sp["candidates"][2],
                         pairs_of_frequent_items=sp["pairs_of_frequent_items"],
                         bitmap_bytes=sp["bitmap_bytes"], pass2_bytes=sp["memory_bytes"][2],
                         apriori_pass2_bytes=4 * sp["pairs_of_frequent_items"]))
    return pd.DataFrame(rows)


# --- Interpreting itemsets: lift, rebel factions, coalitions ----------------------------
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


def rebel_alliances(frequent, n_baskets, party_group):
    """Frequent rebel pairs with lift, labelled same/different group and country."""
    t = to_table(frequent, n_baskets, min_size=2)
    t = t[t["size"] == 2].copy()
    parts = t.itemset.str.split(" + ", regex=False)
    a, b = parts.str[0], parts.str[1]
    t["groups"] = [f"{party_group.get(x, '?')}+{party_group.get(y, '?')}" for x, y in zip(a, b)]
    t["same_group"] = [party_group.get(x) == party_group.get(y) for x, y in zip(a, b)]
    t["same_country"] = a.str.split(":").str[0].to_numpy() == b.str.split(":").str[0].to_numpy()
    return t.sort_values("lift", ascending=False).reset_index(drop=True)


def rebel_pair_summary(baskets, party_group, min_count=30):
    """Who rebels together? Every pair of parties that rebelled together at least
    min_count times (absolute threshold, so small parties are not excluded by support),
    split into: same group / different group & same country / different group & different
    country. Lift 1 = they rebel together exactly as often as if independent."""
    frequent, _ = fi.apriori(baskets, min_count, max_size=2)
    t = rebel_alliances(frequent, len(baskets), party_group)
    t["pair_type"] = np.select(
        [t.same_group, t.same_country],
        ["same group", "different group, same country"],
        "different group, different country")
    summary = (t.groupby("pair_type")
               .agg(pairs=("lift", "size"), co_rebellions=("count", "sum"),
                    median_lift=("lift", "median"), share_lift_above_2=("lift", lambda x: (x > 2).mean()))
               .reset_index())
    summary["share_of_co_rebellions"] = summary.co_rebellions / summary.co_rebellions.sum()
    return summary, t


# --- 2. Group-level coalitions ----------------------------------------------------------
def group_coalitions(d, support=0.3, max_size=4):
    """Frequent group coalitions on adopted contested votes, found with A-priori in each
    term; then the support of every coalition found is computed in BOTH terms (so a
    coalition below the threshold in one term shows its real support, not 0).
    ID (term 9) and PfE (term 10) are both written FAR so the terms can be compared."""
    baskets, found = {}, set()
    for term in (9, 10):
        b, _ = build_baskets(d, term, level="group", passed_only=True)
        baskets[term] = [frozenset("FAR" if g in ("ID", "PFE") else g for g in x) for x in b]
        f, _ = fi.apriori(baskets[term], support, max_size=max_size)
        found |= {s for s in f if len(s) >= 2}
    rows = []
    for s in found:
        row = dict(itemset=" + ".join(sorted(s)), size=len(s))
        for term, b in baskets.items():
            n = len(b)
            sup = sum(s <= x for x in b) / n
            exp = np.prod([sum(i in x for x in b) / n for i in s])
            row[f"support_term{term}"] = sup
            row[f"lift_term{term}"] = sup / exp
        rows.append(row)
    t = pd.DataFrame(rows)
    t["change"] = t.support_term10 - t.support_term9
    return t.sort_values("change").reset_index(drop=True)


# --- Check against HowTheyVote's answer key ---------------------------------------------
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


def validate_against_required(d):
    """Answer key check (term 10): for every adopted vote in groups_required_for_majority,
    is each required group combination inside our group basket (the groups whose
    majority voted FOR)? And how often is each combination present overall?"""
    req, ids = required_coalitions(d)
    b, vids = build_baskets(d, 10, level="group", passed_only=True, vote_ids=ids)
    basket_of = dict(zip(vids, b))
    key = d["required"][d["required"].vote_id.isin(ids)]
    inside = [set(c.split("+")) <= basket_of.get(v, frozenset())
              for v, c in zip(key.vote_id, key.group_codes)]
    req = req.copy()
    req["support_in_our_baskets"] = [
        np.mean([set(c.split("+")) <= x for x in b]) for c in req.required_groups]
    return dict(votes=len(ids), required_rows=len(key),
                share_required_combos_inside_basket=float(np.mean(inside))), req
