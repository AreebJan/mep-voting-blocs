"""Prepares the data. Run from the repo root, with one or more steps in order:

    python src/prepare_data.py download build            # HowTheyVote, terms 9-10
    python src/prepare_data.py group-history votewatch   # VoteWatch, terms 6-8 (optional)

download       fetch the pinned HowTheyVote release into data/raw/ (only missing files)
build          build data/processed/*.parquet from it (~30 s, ~2.5 GB RAM)
group-history  dated group memberships of the VoteWatch MEPs from the EP open-data API
               (~10 min the first time, cached in data/raw/ep_api/)
votewatch      build data/processed/vw_*.parquet; needs the VoteWatch export in
               data/raw/<config.VOTEWATCH_DIR>/ and the group history
"""
import sys

STEPS = ["download", "build", "group-history", "votewatch"]


def run(step):
    if step == "download":
        from data_sources import howtheyvote
        howtheyvote.download()
    elif step == "build":
        from data_sources import howtheyvote
        howtheyvote.build()
    elif step == "group-history":
        from data_sources import ep_api
        ep_api.main()
    elif step == "votewatch":
        from data_sources import votewatch
        votewatch.build()


if __name__ == "__main__":
    steps = sys.argv[1:]
    if not steps or any(s not in STEPS for s in steps):
        print(__doc__)
        sys.exit(1)
    for s in steps:
        run(s)
