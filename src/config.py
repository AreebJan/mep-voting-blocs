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

# --- Older terms: VoteWatch Europe export (2004-2019) ---------------------------------
# HowTheyVote starts on 2019-07-15, so terms 6-8 come from the VoteWatch Europe export
# in data/raw/VOTEWATCH_DIR. Its EP9 file (Jul 2019 - Jun 2022) duplicates HowTheyVote
# and is not used. These terms are kept out of TERMS so the notebook's term 9/10 results
# don't change; load them with data_sources.dataset.load(with_votewatch=True).
VOTEWATCH_DIR = "VoteWatch-EP-voting-data_2004-2022"
VW_TERMS = {
    6: {"rcv": "EP6_RCVs_2022_06_13.xlsx", "docs": "EP6_Voted docs.xlsx",
        "dates": ("2004-07-20", "2009-07-14")},
    7: {"rcv": "EP7_RCVs_2014_06_19.xlsx", "docs": "EP7_Voted docs.xlsx",
        "dates": ("2009-07-14", "2014-07-01")},
    8: {"rcv": "EP8_RCVs_2019_06_25.xlsx", "docs": "EP8_Voted docs.xlsx",
        "dates": ("2014-07-01", "2019-07-02")},
}
# VoteWatch numbers votes 1..N within each term. vote_id = term * offset + N keeps them
# apart from each other and from HowTheyVote's IDs (108425-200503 in this release).
VW_VOTE_ID_OFFSET = 1_000_000
# VoteWatch cell codes -> HowTheyVote position labels. 0 = "not an MEP on that date"
# (row dropped). 4 = absent, 5 = present but didn't vote, 6 = absent with a reason; all
# three are DID_NOT_VOTE in HowTheyVote's coding and are missing in every analysis anyway.
VW_POSITIONS = {1: "FOR", 2: "AGAINST", 3: "ABSTENTION",
                4: "DID_NOT_VOTE", 5: "DID_NOT_VOTE", 6: "DID_NOT_VOTE"}

# The VoteWatch files give each MEP ONE group (the group at export time) with no dates,
# so a group switcher's earlier votes are credited to the wrong group (e.g. all Fidesz
# MEPs show as NI for the whole of EP9, though they sat in EPP until March 2021). We
# instead take each MEP's group on each vote date from the European Parliament's dated
# membership records (src/data_sources/ep_api.py).
EP_API = "https://data.europarl.europa.eu/api/v2/"
# EP abbreviations -> HowTheyVote group codes. Groups that were only renamed keep one
# code across terms (PPE-DE -> EPP, PSE -> SD), following HowTheyVote's own merge of
# GUE_NGL_1995_0 into GUE_NGL. Groups that differ in membership keep their own code:
# ALDE is not Renew (Renew added En Marche in 2019), and the far-right/eurosceptic groups
# of each term (UEN, IND/DEM, ITS, EFD, EFDD, ENF) are different groups.
EP_GROUP_CODES = {
    "PPE-DE": "EPP", "PPE": "EPP", "PSE": "SD", "S&D": "SD",
    "Verts/ALE": "GREEN_EFA", "GUE/NGL": "GUE_NGL", "ECR": "ECR", "NI": "NI",
    "ALDE": "ALDE", "UEN": "UEN", "IND/DEM": "IND_DEM", "ITS": "ITS",
    "EFD": "EFD", "EFDD": "EFDD", "ENF": "ENF",
}
# EP7's file has no EP IDs, so MEPs are matched by name (data_sources/ep_api.py).
# The one MEP whose surname changed completely (married name) is mapped by hand:
# Sidonia Jędrzejewska is listed by the EP as Sidonia MAZUR.
VW_EP7_MANUAL_IDS = {"JĘDRZEJEWSKA": 96782}
# VoteWatch country names that differ from HowTheyVote's countries table
VW_COUNTRY_ALIASES = {"Czech Republic": "Czechia"}

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
