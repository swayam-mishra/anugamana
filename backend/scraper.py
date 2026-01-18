import requests
from bs4 import BeautifulSoup
import json
import time
import re

# ---------------- CONFIG ---------------- #

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

BASE_URL = "https://vedabase.io"
BG_INDEX = "https://vedabase.io/en/library/bg/"

OUTPUT_FILE = "gita_full.json"
REQUEST_DELAY = 1  # seconds (be polite)

# ---------------- HELPERS ---------------- #

def extract_sanskrit(soup):
    container = soup.select_one("div.av-verse_text")
    if not container:
        return ""

    for br in container.find_all("br"):
        br.replace_with("\n")

    text = container.get_text("\n", strip=True)

    ui_phrases = {"Verse text", "Verse Text"}

    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line in ui_phrases:
            continue
        lines.append(line)

    return "\n".join(lines)


def safe_text(soup, selector):
    tag = soup.select_one(selector)
    return tag.get_text(" ", strip=True) if tag else ""


def parse_verse_id(title_text):
    """
    Extract chapter and verse from strings like:
    'Bhagavad-gītā 1.1'
    """
    match = re.search(r"(\d+)\.(\d+)", title_text)
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


# ---------------- MAIN SCRAPER ---------------- #

all_verses = []
seen = set()

for chapter in range(1, 19):
    print(f"\n📘 Scraping Chapter {chapter}")

    chapter_url = f"{BG_INDEX}{chapter}/"
    chapter_resp = requests.get(chapter_url, headers=HEADERS)
    chapter_soup = BeautifulSoup(chapter_resp.text, "html.parser")

    verse_links = [
        a for a in chapter_soup.select("a[href]")
        if re.fullmatch(r"/en/library/bg/\d+/\d+/", a.get("href", ""))
    ]

    for link in verse_links:
        verse_url = BASE_URL + link["href"]

        verse_resp = requests.get(verse_url, headers=HEADERS)
        verse_soup = BeautifulSoup(verse_resp.text, "html.parser")

        title = verse_soup.select_one("h1")
        if not title:
            continue

        ch, vs = parse_verse_id(title.get_text())
        if ch is None:
            continue

        verse_id = f"{ch}.{vs}"
        if verse_id in seen:
            continue
        seen.add(verse_id)

        data = {
            "verse_id": verse_id,
            "chapter": ch,
            "verse": vs,
            "sanskrit": extract_sanskrit(verse_soup),
            "synonyms": safe_text(verse_soup, "div.av-synonyms"),
            "translation": safe_text(verse_soup, "div.av-translation"),
            "purport": safe_text(verse_soup, "div.av-purport"),
        }

        all_verses.append(data)
        print(f"  ✔ {verse_id}")

        time.sleep(REQUEST_DELAY)

# ---------------- SAVE ---------------- #

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(all_verses, f, ensure_ascii=False, indent=2)

print(f"\n DONE. Saved {len(all_verses)} verses to {OUTPUT_FILE}")
