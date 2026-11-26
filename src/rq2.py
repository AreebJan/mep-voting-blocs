"""RQ2: which coalitions pass votes, and did that change after the 2024 election?

Compares A-priori and PCY on FOR and rebel baskets, group coalitions across
terms, and the EPP's role when the left and right blocs disagree."""

import time
import numpy as np
import pandas as pd
from config import POSITIONS, MIN_PARTY_VOTERS
import frequent_items as fi

LEFT_BLOC = ["SD", "GREEN_EFA", "GUE_NGL"]
RIGHT_BLOC = ["ECR", "FAR", "ESN"]           # FAR = ID (term 9) or PfE (term 10)


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


def rebel_alliances(frequent, n_baskets, party_group):
    """Frequent rebel pairs with lift, labelled same/different group and country."""
    t = fi.to_table(frequent, n_baskets, min_size=2)
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


# --- 2. Which national-party coalitions win votes? -------------------------------------
def _far(g):
    """ID (term 9) and PfE (term 10) are both written FAR so the terms can be compared."""
    return "FAR" if g in ("ID", "PFE") else g


def party_groups(d, term):
    """The European group each national party sat in for most of its votes in a term."""
    _, x = _records(d, term)
    x = x[~x.party.str.endswith(":none")]
    return x.groupby("party").group.agg(lambda s: s.mode().iloc[0]).map(_far)


def winning_party_coalitions(d, support=0.5, max_size=3):
    """Our A-priori on the FOR baskets (one per adopted contested vote, items = national
    parties whose majority voted FOR). A frequent itemset is a coalition of parties that
    was on the winning side together on at least `support` of the adopted votes.

    Every itemset gets its countries and European groups, so we can count how many
    winning coalitions cross borders (parties from 2+ countries) and how many cross
    group lines (parties from 2+ European groups). Returns (itemsets, summary, group_mix):
    - itemsets:  one row per frequent coalition of 2..max_size parties
    - summary:   per term and size, how many coalitions and the share crossing borders
                 and groups
    - group_mix: per term, which group combinations the largest coalitions come from"""
    rows = []
    for term in (9, 10):
        baskets, _ = fi.build_baskets(d, term, level="party", passed_only=True)
        frequent, _ = fi.apriori(baskets, support, max_size=max_size)
        group = party_groups(d, term).to_dict()
        n = len(baskets)
        single = {next(iter(s)): c / n for s, c in frequent.items() if len(s) == 1}
        for s, c in frequent.items():
            if len(s) < 2:
                continue
            parties = sorted(s)
            countries = {p.split(":")[0] for p in parties}
            groups = {group.get(p, "NI") for p in parties}
            rows.append(dict(
                term=term, size=len(s), support=c / n,
                lift=(c / n) / np.prod([single[p] for p in parties]),
                coalition=" + ".join(f"{p} ({group.get(p, 'NI')})" for p in parties),
                n_countries=len(countries), groups="+".join(sorted(groups)),
                cross_border=len(countries) > 1, cross_group=len(groups) > 1))
    itemsets = (pd.DataFrame(rows).sort_values(["term", "size", "support"],
                                               ascending=[True, True, False])
                .reset_index(drop=True))
    summary = (itemsets.groupby(["term", "size"])
               .agg(coalitions=("support", "size"), cross_border=("cross_border", "mean"),
                    cross_group=("cross_group", "mean"))
               .reset_index())
    largest = itemsets[itemsets["size"] == max_size]
    group_mix = (largest.groupby(["term", "groups"]).size().rename("coalitions").reset_index())
    group_mix["share"] = group_mix.coalitions / group_mix.groupby("term").coalitions.transform("sum")
    group_mix = (group_mix.pivot_table(index="groups", columns="term", values="share", fill_value=0)
                 .rename(columns=lambda t: f"share_term{t}"))
    group_mix["change"] = group_mix.share_term10 - group_mix.share_term9
    group_mix = group_mix.sort_values("share_term10", ascending=False).reset_index()
    return itemsets, summary, group_mix


def top_coalitions(itemsets, size=2, top=5):
    """Most frequent winning coalitions per term that cross borders, and those that
    also cross group lines."""
    t = itemsets[itemsets["size"] == size]
    cols = ["term", "support", "lift", "coalition"]
    border = t[t.cross_border].groupby("term").head(top)[cols]
    group = t[t.cross_group].groupby("term").head(top)[cols]
    return border, group


# --- 3. Group-level coalitions ----------------------------------------------------------
def group_coalitions(d, support=0.3, max_size=4):
    """Frequent group coalitions on adopted contested votes, found with A-priori in each
    term; then the support of every coalition found is computed in BOTH terms (so a
    coalition below the threshold in one term shows its real support, not 0).
    ID (term 9) and PfE (term 10) are both written FAR so the terms can be compared."""
    baskets, found = {}, set()
    for term in (9, 10):
        b, _ = fi.build_baskets(d, term, level="group", passed_only=True)
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


def validate_against_required(d):
    """Answer key check (term 10): for every adopted vote in groups_required_for_majority,
    is each required group combination inside our group basket (the groups whose
    majority voted FOR)? And how often is each combination present overall?"""
    req, ids = fi.required_coalitions(d)
    b, vids = fi.build_baskets(d, 10, level="group", passed_only=True, vote_ids=ids)
    basket_of = dict(zip(vids, b))
    key = d["required"][d["required"].vote_id.isin(ids)]
    inside = [set(c.split("+")) <= basket_of.get(v, frozenset())
              for v, c in zip(key.vote_id, key.group_codes)]
    req = req.copy()
    req["support_in_our_baskets"] = [
        np.mean([set(c.split("+")) <= x for x in b]) for c in req.required_groups]
    return dict(votes=len(ids), required_rows=len(key),
                share_required_combos_inside_basket=float(np.mean(inside))), req


# --- 4. The EPP as pivot ----------------------------------------------------------------
def _group_majorities(d):
    V, mv = d["votes"], d["member_votes"]
    v = V[(V.term > 0) & V.contested]
    x = mv[mv.vote_id.isin(v.vote_id) & mv.position.isin(POSITIONS)]
    x = pd.DataFrame({"vote_id": x.vote_id.to_numpy(),
                      "group": x.group_code.astype(str).replace({"ID": "FAR", "PFE": "FAR"}).to_numpy(),
                      "pos": np.where(x.position.to_numpy() == "FOR", 1, -1)})
    net = x.groupby(["vote_id", "group"]).pos.sum().unstack(fill_value=0)
    n_for = x[x.pos == 1].groupby(["vote_id", "group"]).size().unstack(fill_value=0)
    n_against = x[x.pos == -1].groupby(["vote_id", "group"]).size().unstack(fill_value=0)
    meta = v.set_index("vote_id")[["term", "timestamp", "passed", "is_main", "procedure_type"]]
    maj = np.sign(net)
    maj["LEFT"] = np.sign(net[[g for g in LEFT_BLOC if g in net]].sum(axis=1))
    maj["RIGHT"] = np.sign(net[[g for g in RIGHT_BLOC if g in net]].sum(axis=1))
    return maj.join(meta), n_for, n_against, meta


def epp_pivot(d):
    """Votes where the left bloc and the right bloc majorities disagree: which side the
    EPP majority takes, and how often the right side wins with / without the EPP.
    Reported for all contested votes, final votes, legislation (COD), and per half-year."""
    maj, _, _, _ = _group_majorities(d)
    s = maj[(maj.LEFT != 0) & (maj.RIGHT != 0) & (maj.LEFT != maj.RIGHT) & (maj.EPP != 0)].copy()
    s["epp_with_right"] = s.EPP == s.RIGHT
    s["right_won"] = np.where(s.RIGHT == 1, s.passed, ~s.passed)
    s["half"] = s.timestamp.dt.year.astype(str) + np.where(s.timestamp.dt.month <= 6, "-H1", "-H2")

    def summarise(df):
        return pd.Series(dict(
            votes=len(df), epp_with_right=df.epp_with_right.mean(),
            right_wins_when_epp_joins=df[df.epp_with_right].right_won.mean(),
            right_wins_when_epp_against=df[~df.epp_with_right].right_won.mean()))
    subsets = {"all contested votes": s, "final votes (is_main)": s[s.is_main],
               "legislation (COD)": s[s.procedure_type == "COD"]}
    by_term = pd.concat({k: v.groupby("term").apply(summarise, include_groups=False)
                         for k, v in subsets.items()}, names=["subset"]).reset_index()
    by_half = s.groupby("half").apply(summarise, include_groups=False).reset_index()
    by_half["term"] = np.where(by_half.half >= "2024-H2", 10, 9)
    return by_term, by_half


def voting_power(d):
    """Share of contested votes that would flip (FOR/AGAINST majority) if one group switched sides,
    ignoring abstentions and special majorities."""
    _, n_for, n_against, meta = _group_majorities(d)
    tot_for, tot_against = n_for.sum(axis=1), n_against.sum(axis=1)
    passed = tot_for > tot_against
    rows = []
    for g in n_for.columns:
        if g == "NI":
            continue
        f2 = tot_for - n_for[g] + n_against[g]
        a2 = tot_against - n_against[g] + n_for[g]
        flipped = (f2 > a2) != passed
        for term in (9, 10):
            ids = meta.index[meta.term == term].intersection(flipped.index)
            members = n_for.loc[ids, g] + n_against.loc[ids, g]
            if members.mean() > 0:
                rows.append(dict(group=g, term=term, pivotal_share=flipped[ids].mean(),
                                 mean_voting_members=members.mean()))
    t = pd.DataFrame(rows).pivot(index="group", columns="term",
                                 values=["pivotal_share", "mean_voting_members"])
    t.columns = [f"{name}_term{term}" for name, term in t.columns]
    return t.reset_index()
