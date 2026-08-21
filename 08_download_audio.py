# 08_download_audio.py
#
# Reads the ddo_entries.json file, finds all unique audio URLs,
# and downloads them into an 'audio' directory. It maintains a JSON map
# for resumability and for use in later steps.

import os
import json
import time
import requests
import logging
from pathlib import Path
from tqdm import tqdm

# --- Configuration ---
INPUT_JSON = "ddo_entries.json"
OUTPUT_DIR = "audio"
MAP_JSON = "audio_map.json"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; AnkiDKDeckDownloader/1.0)"}
MAX_RETRIES = 3
REQUEST_TIMEOUT = 15  # seconds
RETRY_DELAY = 5  # seconds

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def load_json_file(filepath: str, default={}) -> dict:
    """Loads a JSON file, returning a default value if it doesn't exist or is invalid."""
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            logging.warning(f"Could not parse {filepath}, starting fresh.")
    return default


def save_json_file(filepath: str, data: dict):
    """Saves data to a JSON file."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def download_file(url: str, destination: Path) -> bool:
    """Downloads a file from a URL to a destination path with retries."""
    for attempt in range(MAX_RETRIES):
        try:
            with requests.get(
                url, headers=HEADERS, stream=True, timeout=REQUEST_TIMEOUT
            ) as r:
                r.raise_for_status()
                with open(destination, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            return True
        except requests.RequestException as e:
            logging.error(f"Attempt {attempt + 1}/{MAX_RETRIES} failed for {url}: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
    return False


def main():
    """Main function to discover and download audio files."""
    logging.info("--- Starting Audio Downloader ---")

    # 1. Load input data and existing audio map
    if not os.path.exists(INPUT_JSON):
        logging.critical(
            f"Input file '{INPUT_JSON}' not found. Please run 02_generate_entries.py first."
        )
        return

    entries = load_json_file(INPUT_JSON, [])
    audio_map = load_json_file(MAP_JSON)

    # 2. Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 3. Collect all unique audio URLs from the entries
    all_urls = set()
    for entry in entries:
        for udtale in entry.get("udtale", []):
            if audio_url := udtale.get("audio"):
                all_urls.add(audio_url)

    logging.info(f"Found {len(all_urls)} unique audio URLs in total.")

    # 4. Determine which URLs still need to be downloaded
    urls_to_download = [url for url in all_urls if url not in audio_map]

    if not urls_to_download:
        logging.info(
            "All audio files are already downloaded and mapped. Nothing to do."
        )
        logging.info("✔ Audio download complete!")
        return

    logging.info(
        f"{len(audio_map)} files already downloaded. Starting download for {len(urls_to_download)} new files."
    )

    # 5. Download new files with a progress bar
    successful_downloads = 0
    for url in tqdm(urls_to_download, desc="Downloading Audio"):
        # Derive a simple, unique filename from the URL itself
        # e.g., https://.../11024/11024438_1.mp3 -> 11024438_1.mp3
        filename = Path(url).name
        local_path = Path(OUTPUT_DIR) / filename

        if download_file(url, local_path):
            # Success! Update the map and save it immediately.
            audio_map[url] = local_path.as_posix()
            save_json_file(MAP_JSON, audio_map)
            successful_downloads += 1
        else:
            # Failure after retries
            tqdm.write(f"Failed to download {url} after {MAX_RETRIES} attempts.")

    logging.info(f"Downloaded {successful_downloads} new audio files.")
    logging.info("✔ Audio download complete!")


if __name__ == "__main__":
    main()
