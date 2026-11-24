# Downloads the HowTheyVote.eu data export into data/raw/

import urllib.request
from paths import RAW

# Every file in the latest weekly export can be downloaded from this URL
URL = "https://github.com/HowTheyVote/data/releases/latest/download/"

#Tables we need for the project
FILES = ["members", "group_memberships", "groups", "votes", "member_votes",
         "national_parties", "groups_required_for_majority"]


for name in FILES:
    #takes the compressed file
    file = RAW / f"{name}.csv.gz"
    if file.exists():
        print(f"{name} already downloaded")
    else:
        print(f"downloading {name}...")
        urllib.request.urlretrieve(URL + f"{name}.csv.gz", file)

print("finished")