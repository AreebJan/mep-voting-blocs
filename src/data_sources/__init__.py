"""One module per data source, all producing the same tables (see dataset.load):

votes          vote_id, timestamp, term, contested, passed, is_main, count_* ...
member_votes   vote_id, member_id, position, country_code, group_code, party
group_memberships, members (+ vote_topics and required for HowTheyVote)

group_code is the MEP's group ON THE DAY OF THE VOTE; party is "COUNTRY:label".
Named data_sources (not data) so it is not confused with the repo's data/ folder."""
