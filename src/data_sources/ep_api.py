"""Fetches the dated political-group (and national-party) history of every MEP in the
VoteWatch files (terms 6-8) from the European Parliament's open-data API.

Run once from the repo root:   python src/prepare_data.py group-history
Needs data/raw/countries.csv.gz (python src/prepare_data.py download). Takes ~10 min on the
first run; API responses are cached in data/raw/ep_api/, so reruns only fetch what is
missing.

Writes
- data/raw/ep_group_history.csv   member_id, kind, org_id, org_label, start_date, end_date
- data/raw/votewatch_ep7_ids.csv  VoteWatch EP7 row number -> EP website ID

Why: the VoteWatch files store one group per MEP with no dates, so group switchers'
earlier votes are credited to the wrong group (see data_sources/votewatch.py).
"""
import json
import ssl
import time
import unicodedata
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import certifi
import pandas as pd
from paths import RAW
from config import EP_API, VW_TERMS, VW_COUNTRY_ALIASES, VW_EP7_MANUAL_IDS
from data_sources.votewatch import read_meta, EP7_IDS, HISTORY

CACHE = RAW / "ep_api"
KINDS = ("EU_POLITICAL_GROUP", "NATIONAL_POLITICAL_GROUP")
# python.org Python on macOS ships without root certificates, so use certifi's bundle
SSL = ssl.create_default_context(cafile=certifi.where())
PAUSE = 0.3   # seconds between requests; the API returns HTTP 429 when hit faster


def get_json(path, cache_file):
    if cache_file.exists():
        return json.loads(cache_file.read_text())
    url = EP_API + path + ("&" if "?" in path else "?") + "format=application%2Fld%2Bjson"
    req = urllib.request.Request(url, headers={"User-Agent": "mep-voting-blocs/1.0 (research)",
                                               "Accept": "application/ld+json"})
    for attempt in range(8):
        try:
            time.sleep(PAUSE)
            with urllib.request.urlopen(req, timeout=60, context=SSL) as r:
                data = json.loads(r.read())
            break
        except Exception as e:
            if attempt == 7:
                raise RuntimeError(f"{url}: {e}")
            # The API rate-limits (HTTP 429); wait as long as it asks, else back off
            wait = getattr(e, "headers", None) and e.headers.get("Retry-After")
            time.sleep(int(wait) if wait and str(wait).isdigit() else 5 * 2 ** attempt)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(data))
    return data


def norm(s):
    """Lowercase ASCII letters only, so 'ÁDER, János' and 'János Áder' compare equal."""
    s = str(s).translate(str.maketrans({"ø": "o", "Ø": "o", "ł": "l", "Ł": "l", "ß": "ss",
                                        "đ": "d", "Đ": "d", "æ": "ae", "Æ": "ae"}))
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s.lower() if "a" <= c <= "z")


def match_ep7(meta):
    """EP7's file numbers MEPs 1..853 instead of using EP IDs, so match them by name to
    the EP's list of term-7 MEPs: first full name, then surname + first given name, then
    surname alone if it is unique."""
    api = pd.DataFrame(get_json("meps?parliamentary-term=7&offset=0&limit=2000",
                                CACHE / "term7_meps.json")["data"])
    api["member_id"] = api.identifier.astype(int)
    api["k_full"] = (api.givenName + api.familyName).map(norm)
    api["k_first"] = (api.givenName.str.split().str[0] + api.familyName).map(norm)
    api["k_last"] = api.familyName.map(norm)
    meta = meta.assign(k_full=(meta.first_name + meta.last_name).map(norm),
                       k_first=(meta.first_name.str.split().str[0] + meta.last_name).map(norm),
                       k_last=meta.last_name.map(norm))
    meta["member_id"] = pd.NA
    meta["how"] = pd.NA
    for key in ["k_full", "k_first", "k_last"]:
        unique = api.drop_duplicates(key, keep=False).set_index(key)["member_id"]
        todo = meta.member_id.isna()
        meta.loc[todo, "member_id"] = meta.loc[todo, key].map(unique)
        meta.loc[todo & meta.member_id.notna(), "how"] = key

    # Surname variants (extra/married surname, particle, transliteration), e.g. VoteWatch
    # "Alexander ALVARO" = EP "Alexander Nuno PICKART ALVARO", "Valdemar TOMAŠEVSKI" =
    # "Waldemar TOMASZEWSKI": first given name agrees on letters 1-4 or 2-5 (V/Waldemar) and
    # one surname contains the other or both start with the same 5 letters. Only taken
    # if exactly one EP MEP qualifies.
    def similar(a, b):
        return a in b or b in a or a[:5] == b[:5]
    for i in meta.index[meta.member_id.isna()]:
        first, last = norm(meta.first_name[i].split()[0]), norm(meta.last_name[i])
        cand = api[api.givenName.map(lambda g: norm(g.split()[0])[:4] == first[:4]
                                     or norm(g.split()[0])[1:5] == first[1:5])
                   & api.k_last.map(lambda k: similar(k, last))]
        if len(cand) == 1:
            meta.loc[i, ["member_id", "how"]] = [cand.member_id.iloc[0], "surname_variant"]
    for last, mid in VW_EP7_MANUAL_IDS.items():
        meta.loc[meta.last_name.eq(last), ["member_id", "how"]] = [mid, "manual"]

    dup = meta.member_id.dropna().duplicated(keep=False)
    assert not dup.any(), meta.loc[dup[dup].index, ["first_name", "last_name", "member_id"]]
    return meta


def parse(mep):
    rows = []
    for m in mep.get("hasMembership", []):
        kind = str(m.get("membershipClassification", "")).split("/")[-1]
        if kind in KINDS:
            p = m.get("memberDuring", {})
            rows.append({"member_id": int(mep["identifier"]), "kind": kind,
                         "org_id": m["organization"].split("/")[-1],
                         "start_date": p.get("startDate"), "end_date": p.get("endDate")})
    return rows, mep.get("citizenship", "").split("/")[-1]


def main():
    countries = pd.read_csv(RAW / "countries.csv.gz").set_index("label")["code"]
    metas = {t: read_meta(t) for t in VW_TERMS}

    m7 = match_ep7(metas[7])
    print(f"EP7 name matches: {m7.how.value_counts().to_dict()}, "
          f"unmatched: {m7.member_id.isna().sum()} {m7.last_name[m7.member_id.isna()].tolist()}")
    metas[7] = m7.assign(id=m7.member_id).dropna(subset=["id"])

    want = []   # (member_id, expected ISO-3 country)
    for t, meta in metas.items():
        iso = meta.country.replace(VW_COUNTRY_ALIASES).map(countries)
        want += list(zip(meta.id.astype(int), iso, [t] * len(meta)))
    ids = sorted({w[0] for w in want})
    print(f"fetching {len(ids)} MEPs from {EP_API} ...")
    with ThreadPoolExecutor(1) as pool:
        meps = list(pool.map(lambda i: get_json(f"meps/{i}", CACHE / "meps" / f"{i}.json"),
                             ids))
    rows, citizenship = [], {}
    for i, data in zip(ids, meps):
        r, c = parse(data["data"][0])
        rows += r
        citizenship[i] = c
    hist = pd.DataFrame(rows)

    # Check matches by nationality. A mismatch is either a wrong EP7 name match or an MEP
    # elected in a country other than their citizenship (rare but real).
    bad = [(t, i, c, citizenship.get(i)) for i, c, t in want if citizenship.get(i) != c]
    print(f"VoteWatch country != EP citizenship: {len(bad)} {bad[:20]}")

    orgs = sorted(hist.org_id.unique())
    with ThreadPoolExecutor(1) as pool:
        labels = list(pool.map(
            lambda o: get_json(f"corporate-bodies/{o}", CACHE / "orgs" / f"{o}.json")
            ["data"][0].get("label"), orgs))
    hist["org_label"] = hist.org_id.map(dict(zip(orgs, labels)))
    hist = hist[["member_id", "kind", "org_id", "org_label", "start_date", "end_date"]]
    hist.sort_values(["member_id", "kind", "start_date"]).to_csv(HISTORY, index=False)
    m7[["id", "member_id", "how"]].rename(columns={"id": "vw_id"}).dropna(
        subset=["member_id"]).to_csv(EP7_IDS, index=False)
    print(f"wrote {HISTORY.name}: {len(hist):,} memberships, "
          f"group labels: {sorted(hist[hist.kind == KINDS[0]].org_label.unique())}")
