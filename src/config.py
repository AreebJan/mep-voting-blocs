"""Shared parameters for the whole project. Change them here, never inside analysis code,
so every group member's numbers come from the same settings.

Each choice is explained so it can be defended in the report.
"""

# --- Data -----------------------------------------------------------------------------
# HowTheyVote publishes a new release every week. "latest" changes, so we pin one dated
# release to make every number reproducible. All results in the notebook use this one.
RELEASE = "2026-11-21"
BASE_URL = f"https://github.com/HowTheyVote/data/releases/download/{RELEASE}/"

# The data uses an old code for The Left for 3 MEPs in Jul 2019 - Feb 2020.
GROUP_ALIASES = {"GUE_NGL_1995_0": "GUE_NGL"}

# --- Parliamentary terms --------------------------------------------------------------
# Term 9 starts on 1 Feb 2020, after UK MEPs left (Brexit, 31 Jan 2020), so the set of
# MEPs is stable across the term. Term 10 starts at the constitutive session.
TERMS = {
    9: ("2020-02-01", "2024-07-16"),
    10: ("2024-07-16", "2030-01-01"),
}

# --- Which votes and positions count --------------------------------------------------
# Only FOR and AGAINST count as positions. Abstentions and absences are treated as
# missing. Reason: an abstention is ambiguous (protest, conflict of interest, or a
# group line), and FOR = +1 / AGAINST = -1 is the coding used by the SVD map, so the
# graph and the map are built from the same information.
POSITIONS = ("FOR", "AGAINST")

# Drop near-unanimous votes: if more than 90% of FOR/AGAINST votes went one way, the
# vote says almost nothing about blocs and only inflates everyone's agreement.
LOPSIDED = 0.90

# --- Which MEPs count -----------------------------------------------------------------
# MEPs with fewer FOR/AGAINST votes on contested votes are dropped. With too few votes,
# agreement is too noisy to place them (in the 300-vote prototype, every isolated MEP
# had fewer than 50 votes). 500 is about 3% of a term's contested votes.
MIN_VOTES = 500

# Minimum number of votes two MEPs must share before their agreement is trusted.
MIN_SHARED = 100

# --- Agreement graph ------------------------------------------------------------------
# Each MEP is linked to its K most-agreeing colleagues (k-nearest-neighbour backbone).
# A full weighted graph is almost complete (every pair agrees on something) and
# modularity on it mostly reflects the left-right gradient. kNN keeps only the
# strongest ties. No agreement threshold: a threshold isolates MEPs, kNN never does.
K = 15

# --- Louvain --------------------------------------------------------------------------
GAMMA = 1.0                    # resolution; >1 gives more, smaller communities
SEEDS = list(range(10))        # Louvain is order-dependent, so report over 10 seeds
GAMMA_SWEEP = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]

# --- Evaluation -----------------------------------------------------------------------
N_PERMUTATIONS = 200           # shuffled-label baseline for NMI/ARI
MIN_VOTES_POLICY = 150         # minimum contested votes per policy area / per MEP in it

# --- Frequent itemsets (RQ2) ----------------------------------------------------------
# A national party enters a vote's basket if the majority of its MEPs who voted FOR or
# AGAINST voted FOR. Parties with fewer than this many voting MEPs on that vote are
# left out, so one-MEP parties don't add noise.
MIN_PARTY_VOTERS = 2
MIN_SUPPORT = 0.30             # share of baskets an itemset must appear in

# --- RQ3 (group switchers) ------------------------------------------------------------
MIN_PRE_SWITCH_VOTES = 200     # switchers need this many votes before the switch
N_BOOTSTRAP = 2000
