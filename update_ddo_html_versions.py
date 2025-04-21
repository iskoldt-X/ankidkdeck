# Check and download the largest HTML version for each word

import os
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs

HTML_DIR = "ddo_html"  # Your directory for HTML files
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) "
        "Version/18.3.1 Safari/605.1.15"
    )
}
MAX_RETRIES = 5
RETRY_DELAY = 60


def find_alternate_urls(soup, headword):
    """
    Extract all links from searchResultBox whose select parameter begins with <headword>,N
    """
    box = soup.find("div", class_="searchResultBox")
    if not box:
        return []
    urls = []
    for a in box.find_all("a", href=True):
        href = a["href"]
        parsed = urlparse(href)
        qs = parse_qs(parsed.query)
        sel = qs.get("select", [])
        if sel and sel[0].split(",")[0] == headword:
            # Make URL absolute if it's relative
            if href.startswith("/"):
                href = f"https://ordnet.dk{href}"
            urls.append(href)
    # Remove duplicates while preserving order
    return list(dict.fromkeys(urls))


def download_with_retries(url):
    """
    Download with retry logic: on HTTP 503 wait RETRY_DELAY before retrying,
    up to MAX_RETRIES times. Returns (content_bytes or None, error_message or None).
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(url, headers=HEADERS, timeout=10)
            r.raise_for_status()
            return r.content, None
        except requests.exceptions.HTTPError as e:
            status = getattr(e.response, "status_code", None)
            if status == 503 and attempt < MAX_RETRIES:
                print(
                    f"    [!] 503 on {url}, waiting {RETRY_DELAY}s before retry ({attempt}/{MAX_RETRIES})..."
                )
                time.sleep(RETRY_DELAY)
                continue
            return None, f"HTTP {status}: {e}"
        except Exception as e:
            return None, str(e)
    return None, "Max retries reached"


def download_best(urls):
    """
    Always download all urls, compare actual downloaded byte sizes,
    and return (best_url, best_content, best_size).
    """
    best_url = None
    best_content = None
    best_size = -1

    # 1) Probe sizes via HEAD requests and print results
    print("    Candidate version size probing:")
    head_sizes = {}
    for url in urls:
        try:
            r = requests.head(url, headers=HEADERS, timeout=5)
            r.raise_for_status()
            size = int(r.headers.get("Content-Length", 0))
        except Exception:
            size = 0
        head_sizes[url] = size
        print(f"      - {url} → {size} bytes")

    # 2) Download each URL and compare actual size
    print("    Starting download of all candidate versions and comparing actual sizes:")
    for url in urls:
        content, err = download_with_retries(url)
        if content is None:
            print(f"    [!] Download failed: {url} → {err}")
            continue
        size = len(content)
        print(f"    → Download complete {url} ({size} bytes)")
        if size > best_size:
            best_size = size
            best_url = url
            best_content = content

    return best_url, best_content, best_size


def main():
    for fn in sorted(os.listdir(HTML_DIR)):
        if not fn.endswith(".html"):
            continue
        headword = os.path.splitext(fn)[0]
        path = os.path.join(HTML_DIR, fn)

        # Read existing HTML
        with open(path, encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")

        # Find all candidate URLs
        alt_urls = find_alternate_urls(soup, headword)
        if len(alt_urls) <= 1:
            continue  # No extra versions

        print(f"\n[{fn}] Found {len(alt_urls)} versions:")
        best_url, best_content, best_size = download_best(alt_urls)

        if not best_content:
            print("  ↳ No valid download, skipping replacement.")
            continue

        # Backup and replace
        bak_path = path + ".bak"
        os.replace(path, bak_path)
        with open(path, "wb") as out:
            out.write(best_content)
        print(
            f"  ✓ Replaced {fn} with {best_url} ({best_size} bytes), backup saved as {os.path.basename(bak_path)}"
        )

        time.sleep(10)


if __name__ == "__main__":
    main()
