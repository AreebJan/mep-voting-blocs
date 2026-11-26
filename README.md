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
python src/prepare_data.py download build   # CSVs into data/raw/ (release pinned in src/config.py),
                                            # then data/processed/*.parquet (~30 s, ~2.5 GB RAM)
```

Optional, terms 6-8 (2004-2019): put the VoteWatch Europe export in
`data/raw/VoteWatch-EP-voting-data_2004-2022/`, then

```bash
python src/prepare_data.py group-history   # dated group memberships from the EP open-data API (~10 min, cached)
python src/prepare_data.py votewatch       # builds data/processed/vw_*.parquet
```

and load with `data_sources.dataset.load(with_votewatch=True)`. VoteWatch gives each MEP a
single, undated group, so `data_sources/votewatch.py` re-assigns every MEP-vote to the
group the MEP was in on that vote's date. The analysis notebook still uses terms 9 and 10
only.

If you already downloaded the data, run `python src/prepare_data.py download` again: it
only fetches the files that are missing (the policy-area files are new). Raw files from the
2026-11-21 release ("latest" between 21 and 27 Nov) are identical to the pinned ones.

## Run the analysis

Open `analysis.ipynb` and run all cells (about 5 minutes). Every table is written to
`results/tables/` and every figure to `results/figures/`.

Tests for our own implementations (both compare against a reference):

```bash
python tests/test_frequent_items.py   # A-priori and PCY vs brute force
python tests/test_louvain.py          # Louvain vs networkx on known graphs
```

## Layout

The code is organised by method; the notebook and the report by research question.

| Path | What it does |
|---|---|
| `src/config.py` | **All shared parameters** (terms, vote/MEP filters, k, seeds, ...) with the reason for each |
| `src/prepare_data.py` | Command line for the data steps (`download`, `build`, `group-history`, `votewatch`) |
| `src/data_sources/howtheyvote.py` | Terms 9-10: download and cleaning to parquet |
| `src/data_sources/votewatch.py`, `ep_api.py` | Terms 6-8 from VoteWatch, with groups dated per vote from the EP API |
| `src/data_sources/dataset.py` | `load()`: the tables every analysis uses, from one or more sources |
| `src/algorithms/louvain.py` | Our Louvain (Jazmin's, plus a resolution parameter) |
| `src/algorithms/frequent_itemsets.py` | Our A-priori and PCY, plus a brute-force reference |
| `src/algorithms/svd.py` | SVD ideal points (outside technique) |
| `src/analysis/agreement_graph.py` | Sparse MEP x vote matrix, pairwise agreement, kNN graph, pair-agreement table |
| `src/analysis/community_detection.py` | Louvain validation, main comparison, national splits, resolution sweep, policy areas, robustness |
| `src/analysis/evaluation.py` | NMI / AMI / ARI with shuffled-label baselines |
| `src/analysis/ideal_points.py` | Separation on the SVD map, horseshoe check, dimensionality over time |
| `src/analysis/itemset_mining.py` | Baskets, A-priori vs PCY, rebel factions, group coalitions, answer-key check |
| `src/analysis/voting_power.py` | The EPP as pivot, voting power per group |
| `src/analysis/switcher_auc.py` | Did group switchers vote differently before switching? (AUC, cluster bootstrap) |
| `src/figures.py` | Report figures |
| `tests/` | Checks for our Louvain, A-priori and PCY |
| `analysis.ipynb` | Main notebook, runs everything (report appendix) |
| `archive/` | Early prototypes (`Louvain_Attempt_Jazmin.ipynb`, `ep_starter.py`), kept for reference |

| Research question | Modules in `src/analysis/` |
|---|---|
| RQ1: group, country or national party? | `agreement_graph`, `community_detection`, `evaluation` |
| Outside technique: the political map | `ideal_points` |
| RQ2: coalitions | `itemset_mining`, `voting_power` |
| RQ3: group switchers | `switcher_auc` |
