"""Dated political-group and national-party memberships of every MEP from 2004 to now,
from the European Parliament's Open Data API (https://data.europarl.europa.eu, CC BY 4.0).

Covers the MEPs in the VoteWatch files (EP6-EP9) and in HowTheyVote (members.csv.gz).
Run once from the repo root (needs internet; ~20 min the first time, then cached in
data/raw/ep_api/, so reruns only fetch what is missing):
    python src/fetch_ep_history.py

Writes
- data/raw/ep_group_history.csv   member_id, kind, org_id, org_label, start_date, end_date
                                  (kind = EU_POLITICAL_GROUP or NATIONAL_POLITICAL_GROUP)
- data/raw/ep_citizenship.csv     member_id, citizenship (ISO-3), used to check the EP7 names
- data/raw/votewatch_ep7_ids.csv  VoteWatch EP7 row number -> EP ID, and how it was matched

Port of data_sources/ep_api.py on main to this branch's flat layout, extended to the
VoteWatch EP9 and HowTheyVote MEPs so the merged dataset can be checked against the API.
"""
import json
import ssl
import time
import unicodedata
import urllib.request

import pandas as pd
from paths import RAW
from config import EP_API, VW_TERMS, VOTEWATCH_DIR, VW_COUNTRY_ALIASES, VW_EP7_MANUAL_IDS

CACHE = RAW / "ep_api"
HISTORY = RAW / "ep_group_history.csv"
CITIZENSHIP = RAW / "ep_citizenship.csv"
EP7_IDS = RAW / "votewatch_ep7_ids.csv"
KINDS = ("EU_POLITICAL_GROUP", "NATIONAL_POLITICAL_GROUP")
PAUSE = 0.3   # seconds between requests; the API answers HTTP 429 when hit faster


def _ssl():
    # python.org Python on macOS ships without root certificates; use certifi if present
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


SSL = _ssl()


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
            wait = getattr(e, "headers", None) and e.headers.get("Retry-After")
            time.sleep(int(wait) if wait and str(wait).isdigit() else 5 * 2 ** attempt)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(data))
    return data


def read_meta(term):
    """Only the MEP columns of a VoteWatch RCV workbook (fast)."""
    import openpyxl
    path = RAW / VOTEWATCH_DIR / VW_TERMS[term]["rcv"]
    ws = openpyxl.load_workbook(path, read_only=True).worksheets[0]
    rows = list(ws.iter_rows(max_col=11, values_only=True))
    rows = rows[1:] if term == 6 else rows   # EP6: junk header row above the real one
    df = pd.DataFrame(rows[1:], columns=[str(c) for c in rows[0]])
    df = df[df.iloc[:, 0].notna() & (df.iloc[:, 0].astype(str).str.strip() != "")]
    return pd.DataFrame({
        "id": df.iloc[:, 0].astype(str).str.strip().astype(int).to_numpy(),
        "first_name": df["Fname"].astype(str).str.strip().to_numpy(),
        "last_name": df["Lname"].astype(str).str.strip().to_numpy(),
        "country": df["Country"].astype(str).str.strip().to_numpy(),
    })


def norm(s):
    """Lowercase ASCII letters only, so 'ÁDER, János' and 'János Áder' compare equal."""
    s = str(s).translate(str.maketrans({"ø": "o", "Ø": "o", "ł": "l", "Ł": "l", "ß": "ss",
                                        "đ": "d", "Đ": "d", "æ": "ae", "Æ": "ae"}))
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s.lower() if "a" <= c <= "z")


def match_ep7(meta):
    """EP7's file numbers MEPs 1..853 instead of using EP IDs, so match them by name to
    the EP's list of term-7 MEPs: first full name, then surname + first given name, then
    surname alone if it is unique, then surname variants, then the manual list."""
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
    # "Alexander ALVARO" = EP "Alexander Nuno PICKART ALVARO". Taken only if exactly one
    # EP MEP qualifies.
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

    want = []   # (member_id, expected ISO-3 country, term)
    for t, meta in metas.items():
        iso = meta.country.replace(VW_COUNTRY_ALIASES).map(countries)
        want += list(zip(meta.id.astype(int), iso, [t] * len(meta)))
    htv = pd.read_csv(RAW / "members.csv.gz", usecols=["id", "country_code"])
    want += list(zip(htv.id, htv.country_code, ["htv"] * len(htv)))
    ids = sorted({int(w[0]) for w in want})
    todo = sum(not (CACHE / "meps" / f"{i}.json").exists() for i in ids)
    print(f"{len(ids)} MEPs, {todo} not cached yet; fetching from {EP_API} ...")
    rows, citizenship = [], {}
    for n, i in enumerate(ids, 1):
        data = get_json(f"meps/{i}", CACHE / "meps" / f"{i}.json")
        r, c = parse(data["data"][0])
        rows += r
        citizenship[i] = c
        if n % 200 == 0:
            print(f"  {n}/{len(ids)}")
    hist = pd.DataFrame(rows)

    # A country mismatch is either a wrong EP7 name match or an MEP elected in a country
    # other than their citizenship (rare but real)
    bad = sorted({(t, i, c, citizenship.get(i)) for i, c, t in want if citizenship.get(i) != c},
                 key=str)
    print(f"source country != EP citizenship: {len(bad)} {bad[:20]}")

    orgs = sorted(hist.org_id.unique())
    labels = [get_json(f"corporate-bodies/{o}", CACHE / "orgs" / f"{o}.json")["data"][0]
              .get("label") for o in orgs]
    hist["org_label"] = hist.org_id.map(dict(zip(orgs, labels)))
    hist = hist[["member_id", "kind", "org_id", "org_label", "start_date", "end_date"]]
    hist.sort_values(["member_id", "kind", "start_date"]).to_csv(HISTORY, index=False)
    pd.Series(citizenship, name="citizenship").rename_axis("member_id").reset_index().to_csv(
        CITIZENSHIP, index=False)
    m7[["id", "member_id", "how"]].rename(columns={"id": "vw_id"}).dropna(
        subset=["member_id"]).to_csv(EP7_IDS, index=False)
    print(f"wrote {HISTORY.name}: {len(hist):,} memberships of {hist.member_id.nunique()} MEPs")


if __name__ == "__main__":
    main()
