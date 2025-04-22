# Fetch a list of 5000 Danish words from Wiktionary
# NOTE: This script does not include any Den Danske Ordbog content.
# You must download HTML pages manually and comply with their terms of use.


import os
import time
import requests
from urllib.parse import quote
from bs4 import BeautifulSoup


def fetch_danish_wordlist(url):
    """
    Request the specified Wiktionary page and parse the list of Danish words (in page order).
    Returns a list in the format [word1, word2, ...].
    """
    # Get the webpage content
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to fetch page: HTTP {response.status_code}")
        return []

    # Parse the HTML with BeautifulSoup
    soup = BeautifulSoup(response.text, "html.parser")

    # Find the <h3> tag marking the Danish section (id="Danish")
    danish_heading = soup.find("h3", id="Danish")
    if not danish_heading:
        print("Could not find the Danish section heading.")
        return []

    # Get the first ordered list <ol> after the Danish section heading
    word_list_tag = danish_heading.find_next("ol")
    if not word_list_tag:
        print("No word list found in the Danish section.")
        return []

    words = []
    # Iterate through each <li> and extract the text of the <a> tag (the word)
    for li in word_list_tag.find_all("li"):
        a_tag = li.find("a")
        if a_tag:
            word = a_tag.get_text(strip=True)
            words.append(word)

    return words


# Download function


def download_ddo_page(word):
    url = base_url + quote(word)
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        filepath = os.path.join(output_dir, f"{word}.html")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(response.text)
        print(f"[✓] Saved: {word}")
    except Exception as e:
        print(f"[✗] Failed: {word} → {e}")
        with open("download_errors2.log", "a", encoding="utf-8") as errlog:
            errlog.write(f"{word}\t{e}\n")


if __name__ == "__main__":
    url = "https://en.wiktionary.org/wiki/Wiktionary:Frequency_lists/Danish_wordlist"
    wordlist = fetch_danish_wordlist(url)
    if not wordlist:
        print("Failed to retrieve Danish word list.")
        exit(1)
    print(f"Retrieved {len(wordlist)} words.")

    # Output directory
    output_dir = "ddo_html"
    os.makedirs(output_dir, exist_ok=True)

    # Get existing HTML files in the directory (assumes filenames as <word>.html)
    existing_files = os.listdir(output_dir)
    existing_words = set()
    for file in existing_files:
        if file.endswith(".html"):
            # Remove .html extension to get the word
            existing_words.add(file[:-5])

    print(f"Found {len(existing_words)} existing word files.")
    # Filter out words already downloaded
    pending_words = [word for word in wordlist if word not in existing_words]
    print(f"Pending download of {len(pending_words)} words.")

    # Base URL for DDO queries
    base_url = "https://ordnet.dk/ddo/ordbog?query="

    # HTTP headers
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/605.1.15 (KHTML, like Gecko) "
            "Version/18.3.1 Safari/605.1.15"
        )
    }

    # Main loop: only download words not yet downloaded
    for word in pending_words:
        download_ddo_page(word)
        time.sleep(1)  # Recommend not less than 1 second delay

    print("[✓] All words have been downloaded!")
