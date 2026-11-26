"""RQ3: using only votes cast BEFORE a switch, did MEPs who later changed group already
vote differently from the colleagues who stayed?

Data caveat: about half of the rows in group_memberships that look like switches are an
MEP re-entering the SAME group (the membership record is split). Those are dropped.
NI (non-attached) is not a group, so:
- "leaving" test: events whose ORIGIN is a real group (destination may be NI).
  score = disloyalty = share of the MEP's votes against their own group's majority
  (majority computed without the MEP's own vote).
- "joining" test: events whose DESTINATION is a real group (origin may be NI).
  score = agreement with the future group's majority.

For every event the switcher is compared with peers: MEPs in the same origin group over
the same pre-switch window who never left that group during the term. The event's AUC is
P(switcher scores higher than a random peer); the overall AUC is the mean over events
(0.5 = switchers are indistinguishable from stayers). Several MEPs often switch together
(a whole national delegation), so the confidence interval bootstraps over clusters of
(date, origin, destination), not over individual MEPs.
"""
import numpy as np
import pandas as pd
from config import POSITIONS, MIN_PRE_SWITCH_VOTES, N_BOOTSTRAP

MEMBERSHIP_TERMS = {9: ("2019-07-02", "2024-07-16"), 10: ("2024-07-16", "2030-01-01")}
NOT_A_GROUP = {"NI"}


def switch_events(gm):
    gm = gm.sort_values(["member_id", "term", "start_date"]).copy()
    gm["origin"] = gm.groupby(["member_id", "term"]).group_code.shift()
    ev = gm[gm.origin.notna() & (gm.origin != gm.group_code)]
    ev = ev.rename(columns={"group_code": "destination", "start_date": "date"})
    ev = ev[["member_id", "term", "date", "origin", "destination"]].reset_index(drop=True)
    ev["cluster"] = (ev.term.astype(str) + "|" + ev.date.dt.date.astype(str) + "|" +
                     ev.origin + ">" + ev.destination)
    return ev


def _term_records(d, term):
    lo, hi = MEMBERSHIP_TERMS[term]
    v = d["votes"]
    v = v[(v.timestamp >= lo) & (v.timestamp < hi) & v.contested][["vote_id", "timestamp"]]
    mv = d["member_votes"]
    mv = mv[mv.vote_id.isin(v.vote_id) & mv.position.isin(POSITIONS)]
    r = pd.DataFrame({"vote_id": mv.vote_id.to_numpy(), "member_id": mv.member_id.to_numpy(),
                      "group": mv.group_code.astype("string").fillna("NI").to_numpy(),
                      "pos": np.where(mv.position.to_numpy() == "FOR", 1, -1)})
    r = r.merge(v, on="vote_id")
    # net FOR-minus-AGAINST per group per vote, used for group majorities
    net = r.groupby(["vote_id", "group"]).pos.sum().rename("net").reset_index()
    return r, net


def _auc(x, peers):
    peers = np.asarray(peers)
    return (np.mean(x > peers) + 0.5 * np.mean(x == peers)) if len(peers) else np.nan


def run(d, min_votes=MIN_PRE_SWITCH_VOTES, n_boot=N_BOOTSTRAP, seed=0):
    ev = switch_events(d["group_memberships"])
    leavers_by_term = ev.groupby("term").apply(
        lambda e: set(zip(e.member_id, e.origin)), include_groups=False).to_dict()
    rows = []
    for term in sorted(ev.term.unique()):
        r, net = _term_records(d, term)
        left = leavers_by_term.get(term, set())
        for e in ev[ev.term == term].itertuples():
            w = r[(r.timestamp < e.date) & (r.group == e.origin)]
            if w.empty:
                continue
            # peers: same origin group in the window, never left it during this term
            ids = w.member_id.unique()
            peers = [m for m in ids if m != e.member_id and (m, e.origin) not in left]
            w = w[w.member_id.isin(peers + [e.member_id])]
            n = w.groupby("member_id").size()
            ok = n[n >= min_votes].index
            if e.member_id not in ok:
                rows.append(dict(**e._asdict(), status="too few pre-switch votes",
                                 pre_votes=int(n.get(e.member_id, 0))))
                continue
            w = w[w.member_id.isin(ok)]
            row = dict(**e._asdict(), status="ok", pre_votes=int(n[e.member_id]),
                       n_peers=len(ok) - 1)
            # leaving: disagreement with own group's majority, own vote excluded
            if e.origin not in NOT_A_GROUP:
                x = w.merge(net[net.group == e.origin][["vote_id", "net"]], on="vote_id")
                x = x.assign(maj=np.sign(x.net - x.pos))
                x = x[x.maj != 0]
                s = (x.pos != x.maj).groupby(x.member_id).mean()
                row.update(leave_score=s[e.member_id],
                           leave_peer_mean=s.drop(e.member_id).mean(),
                           leave_auc=_auc(s[e.member_id], s.drop(e.member_id)))
            # joining: agreement with the destination group's majority
            if e.destination not in NOT_A_GROUP:
                x = w.merge(net[net.group == e.destination][["vote_id", "net"]], on="vote_id")
                x = x.assign(maj=np.sign(x.net))
                x = x[x.maj != 0]
                s = (x.pos == x.maj).groupby(x.member_id).mean()
                if e.member_id in s.index:
                    row.update(join_score=s[e.member_id],
                               join_peer_mean=s.drop(e.member_id).mean(),
                               join_auc=_auc(s[e.member_id], s.drop(e.member_id)))
            rows.append(row)
    per_event = pd.DataFrame(rows)
    summary = pd.DataFrame([_summarise(per_event, col, n_boot, seed)
                            for col in ["leave_auc", "join_auc"]])
    return ev, per_event, summary


def _summarise(df, col, n_boot, seed):
    x = df.dropna(subset=[col])
    clusters = x.groupby("cluster")[col].apply(list)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(clusters), len(clusters))
        boots.append(np.mean(np.concatenate([clusters.iloc[i] for i in pick])))
    return dict(test=col.replace("_auc", ""), events=len(x), clusters=len(clusters),
                AUC=x[col].mean(), CI_low=np.percentile(boots, 2.5),
                CI_high=np.percentile(boots, 97.5),
                share_events_above_half=(x[col] > 0.5).mean(),
                by_term=x.groupby("term")[col].mean().round(3).to_dict())
