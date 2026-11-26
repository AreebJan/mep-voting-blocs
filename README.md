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
python src/load_votewatch.py   # builds data/processed/ep8_*.parquet (about 30 s)
python src/ep8_analysis.py     # RQ1 + SVD map for EP8, compared with terms 9 and 10
```

Results go to `results/tables/ep8_*.csv`, `results/tables/three_terms_*.csv` and
`results/figures/ep8_svd_map.png`. VoteWatch gives one group per MEP for the whole term,
so group switchers are not handled for EP8 (RQ3 is not run on it).

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
| `src/rq2.py` | RQ2: A-priori vs PCY, rebel factions, group coalitions, EPP as pivot, voting power |
| `src/rq3_switchers.py` | RQ3: did group switchers vote differently before switching? |
| `src/figures.py` | Report figures |
| `tests/test_frequent_items.py` | Checks A-priori / PCY against a brute-force reference |
| `analysis.ipynb` | Main notebook, runs everything (report appendix) |
| `Louvain_Attempt_Jazmin.ipynb`, `ep_starter.py` | Early prototypes, kept for reference |
