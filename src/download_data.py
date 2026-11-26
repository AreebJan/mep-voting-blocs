# Downloads the HowTheyVote.eu data export into data/raw/
# The release is pinned in config.py so every group member gets identical data.

import urllib.request
from paths import RAW
from config import BASE_URL, RELEASE

# Tables we need for the project
FILES = ["members", "group_memberships", "groups", "votes", "member_votes",
         "national_parties", "groups_required_for_majority", "countries",
         "oeil_subjects", "oeil_subject_votes"]

print(f"HowTheyVote release {RELEASE}")
for name in FILES:
    # uses a compressed version of the file for a faster download
    file = RAW / f"{name}.csv.gz"
    if file.exists():
        print(f"{name} already downloaded")
    else:
        print(f"downloading {name}...")
        urllib.request.urlretrieve(BASE_URL + f"{name}.csv.gz", file)

(RAW / "RELEASE.txt").write_text(RELEASE + "\n")
print("finished")
