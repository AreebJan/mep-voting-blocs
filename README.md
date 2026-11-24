# mep-voting-blocs

## Setup

Create a virtual environment and install the requirements:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download the HowTheyVote.eu data and prepare it:

```bash
python src/download_data.py    # downloads the CSVs into data/raw/
python src/cleaned_data.py     # loads the data and member_votes as parquet
```

Make sure to ruun all the commands from the repo root.