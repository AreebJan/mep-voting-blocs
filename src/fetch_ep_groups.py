"""Gets the real EP group history (with dates) of every 2014-2019 MEP from the European
Parliament's Open Data API, so group switchers in the VoteWatch data can be found.

API: https://data.europarl.europa.eu/api/v2 (free, CC BY 4.0, max 500 requests per 5 min).
The MEP ids in the VoteWatch file are the EP's own ids, so they match directly.

Run once from the repo root (takes about 15-20 minutes the first time, needs internet):
    python src/fetch_ep_groups.py
Every API answer is saved in data/raw/votewatch/ep_api/, so if it stops halfway, just run
it again and it continues where it left off.

Writes to data/raw/votewatch/:
    ep8_group_history.csv   one row per MEP per group membership in 2014-2019 (with dates)
    ep8_switches.csv        one row per group change during the term
    ep8_party_history.csv   national party memberships, renamed parties joined into one
After this, run python src/load_votewatch.py again: it then uses the group and national
party the MEP was actually in on the day of each vote.
"""
import json
import time
import urllib.error
import urllib.request
import openpyxl
import pandas as pd
from paths import RAW

API = "https://data.europarl.europa.eu/api/v2"
OUT = RAW / "votewatch"
CACHE = OUT / "ep_api"
TERM_START, TERM_END = "2014-07-01", "2019-07-01"
PAUSE = 0.7   # seconds between requests: about 430 per 5 min, under the 500 limit

# EP group labels in the API -> the codes the rest of the project uses
GROUP_CODES = {
    "PPE": "EPP", "S&D": "SD", "ECR": "ECR", "ALDE": "ALDE", "Verts/ALE": "GREEN_EFA",
    "GUE/NGL": "GUE_NGL", "EFDD": "EFDD", "ENF": "ENF", "NI": "NI",
}


def get(path, name):
    """GET an API path as JSON, cached in a file. Retries when the API says slow down."""
    file = CACHE / f"{name}.json"
    if file.exists():
        return json.loads(file.read_text())
    url = f"{API}/{path}?format=application%2Fld%2Bjson"
    req = urllib.request.Request(url, headers={"User-Agent": "dtu-02807-mep-voting-blocs",
                                               "Accept": "application/ld+json"})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                data = json.loads(r.read())
            file.write_text(json.dumps(data))
            time.sleep(PAUSE)
            return data
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if e.code in (429, 500, 502, 503, 504):
                wait = 30 * (attempt + 1)
                print(f"  API said {e.code}, waiting {wait} s")
                time.sleep(wait)
                continue
            raise
        except urllib.error.URLError:
            time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"gave up on {url}")


def mep_ids():
    """All MEP ids, names and countries in the VoteWatch EP8 file (first columns only)."""
    wb = openpyxl.load_workbook(OUT / "EP8_RCVs_2019_06_25.xlsx", read_only=True)
    rows = wb["Votes"].iter_rows(min_row=2, max_col=5, values_only=True)
    return [(int(r[0]), f"{r[1]} {r[2]}", r[4]) for r in rows if r[0] is not None]


def group_memberships(person, kind="EU_POLITICAL_GROUP"):
    """Memberships of one kind (EP group or national party) that overlap the 2014-2019
    term."""
    out = []
    for m in person.get("hasMembership", []):
        if not str(m.get("membershipClassification", "")).endswith(kind):
            continue
        period = m.get("memberDuring") or {}
        start = period.get("startDate")
        end = period.get("endDate") or TERM_END
        if start is None or start >= TERM_END or end < TERM_START:
            continue
        out.append(dict(org=str(m["organization"]).split("/")[-1],
                        start=max(start, TERM_START), end=min(end, TERM_END)))
    return out


def org_info(org_id, cache={}):
    """(short label, English name) of an EP group or national party."""
    if org_id not in cache:
        body = get(f"corporate-bodies/{org_id}", f"org_{org_id}")
        d = body["data"][0] if body and body.get("data") else {}
        names = d.get("prefLabel") or {}
        if isinstance(names, list):            # list of {"en": ...} objects
            names = {k: v for x in names for k, v in x.items()}
        label = d.get("label") or org_id
        cache[org_id] = (label, names.get("en") or next(iter(names.values()), label))
    return cache[org_id]


def group_label(org_id):
    return org_info(org_id)[0]


INDEPENDENT_WORDS = ("independ", "indépend", "indipend", "unabhäng", "niezale", "bezpartyj",
                     "nezávis", "nezávis", "neodvis", "sans étiquette", "non-party", "nepartin")


def is_independent(label):
    """Labels that mean "no party": these are not joined by name."""
    s = str(label).strip().lower()
    return s in ("", "-", "none") or any(w in s for w in INDEPENDENT_WORDS)


def history_and_switches(rows):
    """Merges back-to-back periods in the same group (e.g. a new role in the same group)
    and lists every change of group."""
    hist = pd.DataFrame(rows).sort_values(["member_id", "start"])
    merged = []
    for _, h in hist.groupby("member_id"):
        cur = None
        for r in h.itertuples(index=False):
            if cur is not None and r.group_code == cur["group_code"]:
                cur["end"] = max(cur["end"], r.end)
                continue
            if cur is not None:
                merged.append(cur)
            cur = r._asdict()
        merged.append(cur)
    hist = pd.DataFrame(merged)
    nxt = hist.groupby("member_id")[["group_code", "start"]].shift(-1)
    m = nxt.group_code.notna()
    sw = hist[m].assign(to_group=nxt.group_code[m].to_numpy(), date=nxt.start[m].to_numpy())
    sw = sw.rename(columns={"group_code": "from_group"})[
        ["member_id", "name", "from_group", "to_group", "date"]]
    return hist, sw.reset_index(drop=True)


def party_history(rows, min_moved=2, min_share=0.8):
    """National party memberships, with ids that are the same party joined into one.

    The EP can give one party several ids:
    - re-registration: the same party name in the same country under another id
      (joined by name; "Independent" and similar are not a party and never joined)
    - renaming (e.g. Front National 4042 -> Rassemblement National 5057 on 12 June 2018):
      joined if, on one day, at least min_share of the old id's members (and at least
      min_moved MEPs) moved to the new id.
    Single MEPs moving to another party are real party changes and stay separate."""
    p = pd.DataFrame(rows).sort_values(["member_id", "start"])
    p["start"], p["end"] = pd.to_datetime(p.start), pd.to_datetime(p.end)
    nxt = p.groupby("member_id")[["org", "start"]].shift(-1)
    moves = p.assign(to=nxt.org, day=nxt.start).dropna(subset=["to"])
    moves = moves[moves.org != moves.to]
    g = moves.groupby(["org", "to", "day"]).member_id.nunique().reset_index(name="moved")
    g["active"] = [p[(p.org == o) & (p.start < d) & (p.end >= d - pd.Timedelta(days=1))]
                   .member_id.nunique() for o, d in zip(g.org, g.day)]
    renames = g[(g.moved >= min_moved) & (g.moved >= min_share * g.active)]

    parent = {}                                  # union-find over party ids
    def root(x):
        while parent.get(x, x) != x:
            x = parent[x]
        return x
    def join(a, b):
        ra, rb = root(a), root(b)
        if ra != rb:
            parent[rb] = ra
    for a, b in zip(renames.org, renames.to):
        join(a, b)
    same_name = 0
    if "name" in p:
        named = p[~p.name.map(is_independent)].drop_duplicates("org")
        for _, ids in named.groupby(["country", named.name.str.strip().str.lower()]).org:
            ids = list(ids)
            same_name += len(ids) - 1
            for b in ids[1:]:
                join(ids[0], b)
    p["party_id"] = p.org.map(root)
    print(f"national parties: {p.org.nunique()} EP ids -> {p.party_id.nunique()} parties "
          f"({len(renames)} renames, {same_name} ids with the same name joined)")
    cols = ["member_id", "party_id", "org", "label", "name", "start", "end"]
    return p[[c for c in cols if c in p]], renames


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    meps = mep_ids()
    rows, party_rows, missing = [], [], []
    for i, (mid, name, country) in enumerate(meps, 1):
        data = get(f"meps/{mid}", f"mep_{mid}")
        if data is None or not data.get("data"):
            missing.append((mid, name))
            continue
        for m in group_memberships(data["data"][0]):
            label = group_label(m["org"])
            rows.append(dict(member_id=mid, name=name, label=label,
                             group_code=GROUP_CODES.get(label, label),
                             start=m["start"], end=m["end"]))
        for m in group_memberships(data["data"][0], "NATIONAL_POLITICAL_GROUP"):
            label, full = org_info(m["org"])
            party_rows.append(dict(member_id=mid, country=country, org=m["org"], label=label,
                                   name=full, start=m["start"], end=m["end"]))
        if i % 50 == 0:
            print(f"{i}/{len(meps)} MEPs done")

    hist, sw = history_and_switches(rows)
    hist.to_csv(OUT / "ep8_group_history.csv", index=False)
    sw.to_csv(OUT / "ep8_switches.csv", index=False)
    parties, _ = party_history(party_rows)
    parties.to_csv(OUT / "ep8_party_history.csv", index=False)

    unknown = sorted(set(hist.label) - set(GROUP_CODES))
    if unknown:
        print("group labels not in GROUP_CODES (add them):", unknown)
    print(f"\n{len(meps)} MEPs, {len(missing)} not found in the API, "
          f"{hist.member_id.nunique()} with a group history")
    print(f"{len(sw)} group changes by {sw.member_id.nunique()} MEPs "
          f"({sw.member_id.nunique() / len(meps):.1%} of MEPs)")
    print("most common moves:")
    print(sw.groupby(["from_group", "to_group"]).size().sort_values(ascending=False)
          .head(10).to_string())


if __name__ == "__main__":
    main()
