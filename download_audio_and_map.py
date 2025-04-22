# NOTE: This script does not include any Den Danske Ordbog content.
# You must download HTML pages manually and comply with their terms of use.


import os
import json
import re
import requests
from urllib.parse import urlparse

# Configuration
INPUT_JSON = "ddo_entries_unique.json"
OUTPUT_DIR = "audio"
MAP_JSON = "audio_map.json"
HEADERS = {"User-Agent": "python-requests/2.x"}


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def derive_filename(resp, url):
    """
    Determine filename from Content-Disposition header if available;
    otherwise use the basename of the URL path.
    Replace spaces in the filename with dots.
    """
    cd = resp.headers.get("content-disposition", "")
    if cd:
        m = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)"?', cd)
        if m:
            raw = m.group(1)
        else:
            raw = os.path.basename(urlparse(url).path)
    else:
        raw = os.path.basename(urlparse(url).path)

    return raw.replace(" ", ".")


def main():
    # 1. Load existing mapping to preserve previously downloaded files
    if os.path.exists(MAP_JSON):
        with open(MAP_JSON, "r", encoding="utf-8") as mf:
            audio_map = json.load(mf)
    else:
        audio_map = {}

    # 2. Read the input entries
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        entries = json.load(f)

    ensure_dir(OUTPUT_DIR)

    # 3. Iterate over entries and download each audio URL
    for entry in entries:
        for ud in entry.get("udtale", []):
            audio_url = ud.get("audio")
            if not audio_url:
                continue

            # Skip URLs already downloaded
            if audio_url in audio_map:
                print(f"Already downloaded, skipping: {audio_url}")
                continue

            try:
                # Perform GET request with streaming to access headers
                resp = requests.get(audio_url, headers=HEADERS, stream=True, timeout=10)
                resp.raise_for_status()

                # Determine local filename
                fname = derive_filename(resp, audio_url)
                local_path = os.path.join(OUTPUT_DIR, fname)

                # If file already exists locally, update map and skip download
                if os.path.exists(local_path):
                    print(f"File exists locally, skipping download: {local_path}")
                    audio_map[audio_url] = local_path
                    resp.close()
                    continue

                # Write content to file in chunks
                with open(local_path, "wb") as wf:
                    for chunk in resp.iter_content(1024 * 8):
                        wf.write(chunk)
                resp.close()

                print(f"Downloaded: {audio_url} → {local_path}")
                audio_map[audio_url] = local_path

            except Exception as e:
                print(f"Failed to download {audio_url}: {e}")

    # 4. Save or update the mapping JSON
    with open(MAP_JSON, "w", encoding="utf-8") as mf:
        json.dump(audio_map, mf, ensure_ascii=False, indent=2)
    print(f"\nSaved mapping to {MAP_JSON}")


if __name__ == "__main__":
    main()
