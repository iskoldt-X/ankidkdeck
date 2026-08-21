# 01_download_all_ddo_versions.py
#
# A robust script to fetch a Danish wordlist from Wiktionary, discover all
# homograph entries for each word on Den Danske Ordbog (DDO), download them
# to uniquely named files, and create a detailed mapping file for future
# processing steps. This version corrects a critical bug in link discovery.

import os
import json
import time
import requests
import random
import logging
from urllib.parse import urlparse, parse_qs, quote, urljoin
from bs4 import BeautifulSoup
from pathlib import Path
from datetime import datetime, timezone

# --- Configuration ---
OUTPUT_DIR = "ddo_html_all_versions"
DDO_BASE_URL = "https://ordnet.dk/"
WORDLIST_CACHE_FILE = "wiktionary_danish_wordlist.txt"
WIKTIONARY_URL = (
    "https://en.wiktionary.org/wiki/Wiktionary:Frequency_lists/Danish_wordlist"
)
MAP_FILE = "download_map.json"
ERROR_LOG_FILE = "download_errors.log"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
}
REQUEST_DELAY_SECONDS = (2, 4)
MAX_RETRIES = 5
RETRY_DELAY_SECONDS = 60
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(), logging.FileHandler("download_perfect.log")],
)


# --- Helper functions ---
def fetch_wiktionary_wordlist(url: str, cache_file: str) -> list[str]:
    if os.path.exists(cache_file):
        logging.info(f"Loading wordlist from cache: {cache_file}")
        with open(cache_file, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    logging.info(f"Fetching wordlist from URL: {url}")
    try:
        response = requests.get(url, headers=HEADERS, timeout=20)
        response.raise_for_status()
    except requests.RequestException as e:
        logging.error(f"Failed to fetch Wiktionary page: {e}")
        return []
    soup = BeautifulSoup(response.text, "html.parser")
    danish_heading = soup.find("h3", id="Danish")
    if not danish_heading or not (word_list_tag := danish_heading.find_next("ol")):
        logging.error("Could not parse Wiktionary wordlist.")
        return []
    words = [
        a.get_text(strip=True)
        for li in word_list_tag.find_all("li")
        if (a := li.find("a"))
    ]
    with open(cache_file, "w", encoding="utf-8") as f:
        for word in words:
            f.write(f"{word}\n")
    logging.info(f"Successfully fetched and cached {len(words)} words.")
    return words


def load_map_file(filepath: str) -> dict:
    if os.path.exists(filepath):
        logging.info(f"Loading existing map file from {filepath}")
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            logging.error(
                f"Could not decode JSON from {filepath}. Starting with an empty map."
            )
            return {}
    return {}


def save_map_file(filepath: str, data: dict):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def download_url(url: str) -> requests.Response | None:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            time.sleep(random.uniform(*REQUEST_DELAY_SECONDS))
            response = requests.get(url, headers=HEADERS, timeout=20)
            if response.status_code == 404:
                logging.warning(f"URL not found (404): {url}")
                return None
            response.raise_for_status()
            return response
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 503 and attempt < MAX_RETRIES:
                logging.warning(
                    f"Server busy (503). Waiting {RETRY_DELAY_SECONDS}s... ({attempt}/{MAX_RETRIES})"
                )
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                logging.error(f"HTTP Error for {url}: {e}")
                return None
        except requests.RequestException as e:
            logging.error(f"Request failed for {url}: {e}")
    return None


def generate_filename_from_url(url: str, word: str, index: int) -> str:
    try:
        query_params = parse_qs(urlparse(url).query)
        select_val = query_params.get("select", [None])[0]
        if select_val:
            parts = select_val.lower().split(",")
            base = parts[0].replace(" ", "_")
            idx = parts[1] if len(parts) > 1 else "0"
            return f"{base}__{idx}.html"
    except Exception:
        pass
    return f"{word.lower()}__{index}.html"


def log_error_entry(word: str, rank: int, message: str):
    logging.error(f"[{word}] {message}")
    with open(ERROR_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(
            f"{datetime.now(timezone.utc).isoformat()}\t{rank}\t{word}\t{message}\n"
        )


# --- Main ---
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logging.info("--- Starting Perfect DDO Downloader (v3 - Corrected) ---")

    wordlist = fetch_wiktionary_wordlist(WIKTIONARY_URL, WORDLIST_CACHE_FILE)
    if not wordlist:
        logging.critical("Could not retrieve word list. Aborting.")
        return

    download_map = load_map_file(MAP_FILE)
    processed_words = {entry["query_word"] for entry in download_map.values()}

    total_words = len(wordlist)
    logging.info(
        f"Wordlist contains {total_words} words. {len(processed_words)} words already processed."
    )

    for rank, word in enumerate(wordlist, 1):
        if word in processed_words:
            continue

        logging.info(f"--- Processing word {rank}/{total_words}: {word} ---")
        search_url = f"{urljoin(DDO_BASE_URL, '/ddo/ordbog?query=')}{quote(word)}"
        search_page_response = download_url(search_url)

        if not search_page_response:
            log_error_entry(word, rank, f"Failed to download search page: {search_url}")
            continue

        soup = BeautifulSoup(search_page_response.text, "html.parser")
        urls_to_download = set()

        # === START OF CRITICAL FIX ===
        # Use the simple, robust logic from the old script.
        # Find ALL searchResultBox divs, not just ones inside opslagsordBox.
        all_result_boxes = soup.find_all("div", class_="searchResultBox")
        for box in all_result_boxes:
            links = box.find_all("a", href=True)
            for a in links:
                href = a["href"]
                # Check if the link is for a headword entry, not a fixed expression (mselect)
                if "select=" in href and "mselect=" not in href:
                    query_params = parse_qs(urlparse(href).query)
                    if (
                        select_param := query_params.get("select", [None])[0]
                    ) and select_param.lower().split(",")[0] == word.lower():
                        urls_to_download.add(urljoin(DDO_BASE_URL, href))
        # === END OF CRITICAL FIX ===

        if not urls_to_download:
            # If no suitable links found, it's a direct landing or single result
            urls_to_download.add(search_page_response.url)

        logging.info(f"[{word}] Found {len(urls_to_download)} version(s) to download.")

        # The rest of the loop remains the same, it was already correct.
        for i, url in enumerate(sorted(list(urls_to_download))):
            filename = generate_filename_from_url(url, word, i)
            filepath = Path(OUTPUT_DIR) / filename
            if filepath.exists() and filename in download_map:
                logging.info(
                    f"[{word}] -> SKIPPING {filename}, already exists and mapped."
                )
                continue

            logging.info(
                f"[{word}] -> DOWNLOADING version {i+1}/{len(urls_to_download)}: {url} -> {filename}"
            )
            entry_response = download_url(url)
            if entry_response:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(entry_response.text)
                download_map[filename] = {
                    "query_word": word,
                    "query_rank": rank,
                    "source_url": search_url,
                    "final_url": entry_response.url,
                    "download_timestamp": datetime.now(timezone.utc).isoformat(),
                }
                save_map_file(MAP_FILE, download_map)
                logging.info(f"[{word}] -> SAVED {filename} and updated map.")
            else:
                log_error_entry(word, rank, f"Failed to download entry page: {url}")

        processed_words.add(word)

    logging.info("--- Download process complete! ---")


if __name__ == "__main__":
    choice = input(
        f"WARNING: You are about to run the downloader.\n"
        f"The script will try to resume from '{MAP_FILE}'.\n"
        f"Do you want to clear all progress ({OUTPUT_DIR} and {MAP_FILE}) for a completely fresh start? (y/N): "
    )
    if choice.lower() == "y":
        import shutil

        logging.info("Clearing old data for a fresh start...")
        if os.path.exists(OUTPUT_DIR):
            shutil.rmtree(OUTPUT_DIR)
            logging.info(f"Removed old directory: {OUTPUT_DIR}")
        if os.path.exists(MAP_FILE):
            os.remove(MAP_FILE)
            logging.info(f"Removed old map file: {MAP_FILE}")
        logging.info("Old data cleared. Starting fresh download.")
    else:
        logging.info("Resuming download. No data will be deleted.")

    main()
