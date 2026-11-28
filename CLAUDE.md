# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

DTU 02807 (Computational Tools for Data Science) final project: do European Parliament MEPs vote with their country, their national party, or their European political group? Three research questions:
- **RQ1**: do Louvain communities on an MEP agreement graph follow group, country or national party? (`src/rq1.py`, plus SVD ideal-point map in `src/svd_map.py`)
- **RQ2**: which national-party coalitions win votes / rebel together, found with frequent itemsets (`src/rq2.py`, `src/frequent_items.py`)
- **RQ3**: did group switchers vote differently before switching? (`src/rq3_switchers.py`)

## Commands

All commands run from the repo root, inside the venv (`python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`).

```bash
python src/download_data.py    # HowTheyVote CSVs -> data/raw/ (only fetches missing files)
python src/cleaned_data.py     # -> data/processed/*.parquet (~30 s, ~2.5 GB RAM)
python tests/test_frequent_items.py   # or: pytest tests/
pytest tests/test_frequent_items.py::test_apriori_toy   # single test
```

The main analysis is `analysis.ipynb` (run all, ~4 min); it writes every table to `results/tables/` and every figure to `results/figures/`. These outputs are committed.

Extra 2014-19 term (EP8, VoteWatch data; needs `EP8_RCVs_2019_06_25.xlsx` and `EP8_Voted_docs.xlsx` in `data/raw/votewatch/`):

```bash
python src/fetch_ep_groups.py  # optional, internet, ~15 min: per-date group/party history from EP Open Data API
python src/load_votewatch.py   # -> data/processed/ep8_*.parquet
python src/ep8_analysis.py     # RQ1 + SVD for EP8 vs terms 9 and 10 -> ep8_*.csv, three_terms_*.csv
```

Merged 2004-2026 dataset (VoteWatch EP6-8 + HowTheyVote + EP API dated memberships; VoteWatch export in `data/raw/VoteWatch-EP-voting-data_2004-2022/`):

```bash
python src/fetch_ep_history.py  # EP API -> data/raw/ep_group_history.csv, votewatch_ep7_ids.csv, ep_citizenship.csv
python src/build_merged.py      # -> data/processed/merged_votes.parquet, merged_member_votes.parquet, vw9_*.parquet
python src/verify_merged.py     # 44 checks, exits 1 on failure; run after any change to build_merged.py
```

Python 3.12 is needed (the pinned requirements don't install on 3.11).

## Architecture

- **Flat imports from `src/`.** Modules import each other as `from config import ...`, `from agreement import ...` (no package). Scripts work because `python src/x.py` puts `src/` on `sys.path`; the notebook and tests insert `src/` into `sys.path` themselves. Keep this convention.
- **`src/config.py` holds every shared parameter** (pinned data release, term date ranges, `LOPSIDED`, `MIN_VOTES`, `MIN_SHARED`, `K`, `GAMMA`, `SEEDS`, itemset supports, ...), each with a written justification for the report. Change parameters there, never inside analysis code, and keep the justification comments up to date.
- **`src/paths.py`** defines `ROOT`, `RAW` (`data/raw`), `PROC` (`data/processed`). `data/` is gitignored.
- **Data flow:** raw CSVs → `cleaned_data.build()` → `votes.parquet`, `member_votes.parquet` (~18M rows, compact dtypes), `vote_topics.parquet`. Everything downstream loads via `from cleaned_data import load; d = load()` and passes the dict `d` around.
- **RQ1 pipeline:** `agreement.build_term_graph(d, term)` → sparse MEP×vote matrix (FOR=+1, AGAINST=−1), pairwise agreement (pairs need `MIN_SHARED` shared votes), kNN backbone graph `W`. `louvain.louvain_from_scratch(W, seed, gamma)` is our own implementation (validated against networkx in `rq1.validate_louvain`); results reported as best-modularity over `SEEDS`. `evaluation.py` compares partitions with group/country/party labels via NMI/AMI/ARI against shuffled-label baselines.
- **RQ2:** `frequent_items.py` has our own A-priori and PCY plus a `brute_force` reference that the tests check against. Baskets = national parties whose majority voted FOR on a vote (parties with < `MIN_PARTY_VOTERS` voters excluded).
- **Merged dataset (`build_merged.py`):** VoteWatch is wide (one row per MEP, one column per vote, codes 0-6) and is melted to long. VoteWatch vote ids become `term * 1_000_000 + n`; HowTheyVote ids are kept. Group/party for 2004-19 come from `ep_group_history.csv` on the vote date (`on_date`), VoteWatch's static end-of-term label is only a fallback. VoteWatch date cells mix Excel dates (correct as stored) and `dd.mm.yyyy` text: never parse them all as text with `dayfirst=True` (that flips the Excel ones). Settings live in config.py's "Merged 2004-2026 dataset" section.
- `figures.py` makes the report figures; analysis modules return DataFrames and the notebook saves them.

## Data conventions worth knowing

- Only FOR/AGAINST count; abstentions and absences are missing. Near-unanimous votes (> `LOPSIDED` one way) are dropped as non-contested.
- Term 9 starts 2020-02-01 (post-Brexit) not at the 2019 election; term 10 from 2024-07-16. `term == 0` means outside both study periods.
- Party key is `"COUNTRY:short_label"` (national party IDs change between terms). `GROUP_ALIASES` maps an old Left group code.
- `passed` uses `result` where it exists (2024+), otherwise approximated as FOR > AGAINST.
- VoteWatch EP8: dates have day/month swapped when day ≤ 12 (fixed in the loader); only one end-of-term group/party per MEP unless the EP-API history file exists, so 2014-19 national-party results are left out of the three-term comparison.
- Louvain, A-priori and PCY are the project's own implementations; networkx and `brute_force` serve only as validation references, so don't swap the implementations for library calls.
- `Louvain_Attempt_Jazmin.ipynb` and `ep_starter.py` are early prototypes kept for reference, not part of the pipeline.
