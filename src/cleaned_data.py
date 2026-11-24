import pandas as pd
from paths import RAW, PROC

# loading data
members = pd.read_csv(RAW / "members.csv.gz", usecols=["id", "first_name", "last_name", "country_code"])
gm      = pd.read_csv(RAW / "group_memberships.csv.gz", parse_dates=["start_date", "end_date"])
groups  = pd.read_csv(RAW / "groups.csv.gz")
votes   = pd.read_csv(RAW / "votes.csv.gz",
                      usecols=["id", "timestamp", "is_main", "procedure_type",
                               "count_for", "count_against", "result"],
                      parse_dates=["timestamp"])

# convert the member votes to a parquet file and then loading the data 
mv_path = PROC / "member_votes.parquet"

# Only reads the csv once, and after that it will load much faster using parquet
# add dtype to keep the memory usage a lot smaller
if not mv_path.exists():
    mv = pd.read_csv(RAW / "member_votes.csv.gz",
                     usecols=["vote_id", "member_id", "position", "country_code",
                              "group_code", "national_party_id"],
                     dtype={"vote_id": "int32", "member_id": "int32",
                            "position": "category", "country_code": "category",
                            "group_code": "category", "national_party_id": "category"})
    mv.to_parquet(mv_path)

mv = pd.read_parquet(mv_path)
print(mv.shape)