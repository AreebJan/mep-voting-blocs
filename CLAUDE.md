# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

DTU 02807 (Computational Tools for Data Science) group project: do European Parliament MEPs vote with their country or their party? Uses HowTheyVote.eu roll-call data for parliamentary terms 9 and 10. Research questions: RQ1 (Louvain communities vs. groups/countries), RQ2 (frequent itemsets of national parties voting FOR together), RQ3 (did group switchers vote differently before switching). Plus an SVD ideal-point map, the technique from outside the course.

## Commands

Run everything from the repo root (with `.venv` activated).

```bash
pip install -r requirements.txt
python src/prepare_data.py download build   # fetch CSVs into data/raw/ (only missing files), then build data/processed/*.parquet (~30 s, ~2.5 GB RAM)
python src/prepare_data.py group-history votewatch   # optional: terms 6-8 from VoteWatch (needs the export in data/raw/)
python tests/test_frequent_items.py   # or: pytest tests/
python tests/test_louvain.py
pytest tests/test_frequent_items.py::test_apriori_toy   # single test
```

The full analysis is `analysis.ipynb` (run all cells, ~5 min). It writes every table to `results/tables/` and every figure to `results/figures/`, and those outputs are committed. `data/` is gitignored.

## Architecture

- **Layout by method.** `src/algorithms/` = our own implementations (numpy only), `src/analysis/` = one module per method or technique applied to the data, `src/data_sources/` = one module per data source (named so, not `data`, to avoid clashing with the repo's `data/` folder). `config.py`, `paths.py`, `figures.py` and the `prepare_data.py` command line stay at `src/` root. The notebook (and the report) are organised by research question instead; the table in README.md maps one to the other.
- **Imports are absolute from `src/`** (`from config import K`, `from paths import RAW`, `from algorithms.louvain import modularity`). The notebook and tests do `sys.path.insert(0, "src")` first; `src/prepare_data.py` runs from `src/` so the same imports work. Keep that style; don't add relative imports.
- **`src/config.py` is the only place for parameters**: pinned data release, term dates, vote/MEP filters, kNN `K`, Louvain `GAMMA`/`SEEDS`, support thresholds, bootstrap counts. Each one has a comment justifying it for the report. Don't hardcode these values in analysis code. If you add a parameter, put it in config.py with its reason.
- **Data flow**: `data_sources.dataset.load()` (optionally `with_votewatch=True` for terms 6-8) returns a dict of DataFrames (`votes`, `member_votes`, `vote_topics`, `group_memberships`, `members`, `required`) and builds the parquet files on first use. `votes` has derived `term` (0 = outside both terms), `contested` (top side ≤ `LOPSIDED`) and `passed` columns. `member_votes` has a `party` key of the form `"COUNTRY:short_label"`, because national party IDs change between terms.
- **Shared conventions**: only FOR/AGAINST count as positions (+1/−1). Abstentions and absences are missing. Most analyses use contested votes only, and only MEPs with at least `MIN_VOTES` such votes.
- **RQ1 pipeline**: `analysis.agreement_graph.build_term_graph(d, term)` → sparse MEP × vote matrix → pairwise agreement (needs `MIN_SHARED` shared votes) → symmetric kNN graph `W`. Then `algorithms.louvain.louvain_from_scratch(W, seed, gamma)` is run over `SEEDS`, and `analysis.evaluation.compare` scores the result as NMI/AMI/ARI against group/country/party labels, always next to a shuffled-label baseline. `analysis/community_detection.py` orchestrates this.
- **Own implementations are part of the assignment**: Louvain (`algorithms/louvain.py`, originally from `archive/Louvain_Attempt_Jazmin.ipynb`) and A-priori/PCY (`algorithms/frequent_itemsets.py`) are written from scratch. networkx Louvain is used only to validate them. Don't swap in library versions. If you change either, rerun the tests (`tests/test_louvain.py` compares with networkx, `tests/test_frequent_items.py` with `brute_force`).
- **Analysis modules** (`community_detection`, `ideal_points`, `itemset_mining`, `voting_power`, `switcher_auc`, ...) expose functions that return DataFrames. The notebook calls them, saves the CSVs and passes the results to `figures.py`.
- `archive/ep_starter.py` and `archive/Louvain_Attempt_Jazmin.ipynb` are early prototypes, kept for reference only.
