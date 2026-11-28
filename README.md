# mep-voting-blocs

European Parliament: do MEPs vote with their country or with their party?
DTU 02807 Computational Tools for Data Science, final project.

## Setup

Create a virtual environment and install the requirements:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download the HowTheyVote.eu data and prepare it (run all commands from the repo root):

```bash
python src/download_data.py    # downloads the CSVs into data/raw/ (release pinned in src/config.py)
python src/cleaned_data.py     # builds data/processed/*.parquet (about 30 s, ~2.5 GB RAM)
```

If you already downloaded the data, run `python src/download_data.py` again: it only
fetches the files that are missing (the policy-area files are new). Raw files from the
2026-11-21 release ("latest" between 21 and 27 Nov) are identical to the pinned ones.

## Run the analysis

Open `analysis.ipynb` and run all cells (about 4 minutes). Every table is written to
`results/tables/` and every figure to `results/figures/`.

Tests for the frequent-itemset code:

```bash
python tests/test_frequent_items.py
```

## Extra term: 2014-2019 (VoteWatch data)

To check that the RQ1 result also holds before 2019, the 2014-2019 term (EP8) is loaded
from the VoteWatch Europe dataset (EUI, https://cadmus.eui.eu/handle/1814/74918).
Put `EP8_RCVs_2019_06_25.xlsx` and `EP8_Voted_docs.xlsx` in `data/raw/votewatch/`, then:

```bash
python src/fetch_ep_groups.py  # optional, needs internet: real group history from the EP API (~15 min)
python src/load_votewatch.py   # builds data/processed/ep8_*.parquet (about 30 s)
python src/ep8_analysis.py     # RQ1 + SVD map for EP8, compared with terms 9 and 10
```

Results go to `results/tables/ep8_*.csv`, `results/tables/three_terms_*.csv` and
`results/figures/ep8_svd_map.png` and `results/figures/three_terms_rq1.png`. VoteWatch gives one group per MEP for the whole term
(the one at the end of the term). `fetch_ep_groups.py` gets each MEP's real group and national party
history with dates from the EP Open Data API (https://data.europarl.europa.eu, CC BY 4.0) and writes
`data/raw/votewatch/ep8_switches.csv`; `load_votewatch.py` then uses the group and party the MEP
had on the day of each vote (renamed parties, e.g. Front National -> Rassemblement National,
count as one party).
For RQ1 the comparison is 2014-19 vs 2019-24 and 2024-26 side by side
(`results/tables/three_terms_summary.csv`). National-party results are shown for 2019-2026
only: VoteWatch gives one end-of-term party label per MEP, so the 2014-19 party numbers are
left out unless `ep8_party_history.csv` is present. The loader also fixes the VoteWatch dates, which have day and month
swapped whenever the day is 12 or less.

## Merged dataset 2004-2026 (VoteWatch + HowTheyVote + EP API)

One table of every roll-call vote from Sept 2004 to now: VoteWatch EP6-EP8 (2004-2019)
stacked with HowTheyVote (from July 2019), with each MEP's group and national party taken
from the EP's dated membership records for the day of each vote. Put the VoteWatch export
in `data/raw/VoteWatch-EP-voting-data_2004-2022/` (EP6-EP9 `*_RCVs_*.xlsx` and
`*_Voted docs.xlsx`), run the HowTheyVote steps above, then:

```bash
python src/fetch_ep_history.py  # needs internet, ~20 min first time (cached in data/raw/ep_api/)
python src/build_merged.py      # data/processed/merged_*.parquet (~3 min, ~6 GB RAM)
python src/verify_merged.py     # 44 checks -> results/tables/merged_verification.csv
```

Load with `from build_merged import load_merged; d = load_merged()`. `member_id` is the
EP's own ID in every term (EP7's file has none, so its MEPs are matched by name). VoteWatch
EP9 (2019-2022) is not in the merged data; `verify_merged.py` uses it to check that the
same build steps reproduce HowTheyVote's positions, groups and parties on 2019-2022.
Notes: `term` follows the legislatures (term 9 starts July 2019, unlike `config.TERMS`);
VoteWatch "absent" rows for periods outside an MEP's seat are dropped; a few VoteWatch date
typos are repaired (printed during the build); `party` labels are the EP's short labels, so
a renamed or re-registered party can change label between terms (`national_party_id` is
the EP's id for one term).

### The analysis on all five terms (2004-2026)

After `build_merged.py`, these rerun the notebook's pipeline on terms 6-10 with the same
settings and code. Each one checks that its terms 9-10 results equal the notebook's saved
tables, and writes `results/tables/all_terms_*.csv` and `results/figures/all_terms_*.png`:

```bash
python src/all_terms_analysis.py   # RQ1 + SVD map
python src/all_terms_rq2.py        # RQ2 (right bloc = every group to the EPP's right, per term)
python src/all_terms_rq3.py        # RQ3 (group dissolutions left out; no stayers to compare)
python src/all_terms_rest.py       # overview, Louvain vs networkx, policy areas, extremes, dimensionality
```

Before 2019, policy areas are VoteWatch's own (21 areas), not OEIL subjects, and the
"extremes" check uses ALDE in the centre and the right bloc without ECR as the far right.

## Layout

| Path | What it does |
|---|---|
| `src/config.py` | **All shared parameters** (terms, vote/MEP filters, k, seeds, ...) with the reason for each |
| `src/download_data.py`, `src/cleaned_data.py` | Data download and cleaning to parquet |
| `src/agreement.py` | Sparse MEP x vote matrix, pairwise agreement, kNN graph |
| `src/louvain.py` | Our Louvain implementation (Jazmin's, plus a resolution parameter) |
| `src/evaluation.py` | NMI / AMI / ARI with shuffled-label baselines |
| `src/rq1.py` | RQ1: validation, main comparison, pair agreement, national splits, resolution sweep, policy areas, robustness |
| `src/svd_map.py` | Outside technique: SVD ideal-point map, horseshoe check, dimensionality over time |
| `src/frequent_items.py` | Our A-priori and PCY, baskets, brute-force reference |
| `src/rq2.py` | RQ2: A-priori vs PCY, winning national-party coalitions (cross-border / cross-group), rebel factions, group coalitions, EPP as pivot, voting power |
| `src/rq3_switchers.py` | RQ3: did group switchers vote differently before switching? |
| `src/figures.py` | Report figures |
| `tests/test_frequent_items.py` | Checks A-priori / PCY against a brute-force reference |
| `analysis.ipynb` | Main notebook, runs everything (report appendix) |
| `Louvain_Attempt_Jazmin.ipynb`, `ep_starter.py` | Early prototypes, kept for reference |
