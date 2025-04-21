# Remove duplicate entries by selecting the one with the highest richness score

import json
from collections import defaultdict


def richness_score(entry):
    """
    Calculate a "richness" score for an entry by summing counts of its main fields.
    """
    score = 0
    score += len(entry.get("definitions", []))
    score += len(entry.get("fixed_expressions", []))
    score += len(entry.get("wordforms", []))
    score += len(entry.get("udtale", []))
    # orddannelser is a dict: include counts from each category
    for lst in entry.get("orddannelser", {}).values():
        score += len(lst)
    return score


# 1. Load original data
with open("ddo_entries.json", "r", encoding="utf-8") as f:
    entries = json.load(f)

# 2. Group entries by headword
groups = defaultdict(list)
for entry in entries:
    hw = entry.get("headword")
    groups[hw].append(entry)

# 3. Compute statistics
total_entries = len(entries)
unique_headwords = len(groups)
duplicates = total_entries - unique_headwords

print(f"Total entries: {total_entries}")
print(f"Unique headwords: {unique_headwords}")
print(f"Duplicate entries: {duplicates}\n")
print("Duplicate headwords and their counts:")
for hw, lst in groups.items():
    if len(lst) > 1:
        print(f"  {hw}: {len(lst)}")

# 4. Build deduplicated list: choose the entry with the highest richness score from each group
unique_entries = []
for hw, lst in groups.items():
    # Select the entry with the max richness_score (first one if tied)
    best = max(lst, key=richness_score)
    unique_entries.append(best)

# 5. Write the deduplicated output file
with open("ddo_entries_unique.json", "w", encoding="utf-8") as f:
    json.dump(unique_entries, f, ensure_ascii=False, indent=2)

print("\n✔ Deduplication complete. File saved as 'ddo_entries_unique.json'.")
