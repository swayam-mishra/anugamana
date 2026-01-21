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

def clean_label(text, label):
    """
    Removes a label (e.g., 'Translation', 'Purport') from the start of the text.
    Handles variations like 'Translation:', 'Translation ' etc.
    Case-insensitive match for the prefix.
    """
    if not text:
        return ""
    # Regex: Start of string + optional space + label + optional colon + whitespace
    # e.g. matches "Translation", "Translation:", "Translation "
    pattern = rf"^\s*{label}[:\s]*"
    return re.sub(pattern, "", text, flags=re.IGNORECASE).strip()

def extract_sanskrit(soup):
    container = soup.select_one("div.av-verse_text")
    if not container:
        return ""

    # Create a copy to avoid modifying the original soup validation
    import copy
    container = copy.copy(container)

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
    if tag:
        # Use \n\n for paragraph breaks to maintain readability
        return tag.get_text("\n\n", strip=True)
    return ""

def extract_purport_robust(soup):
    """
    Tries multiple strategies to find the purport.
    """
    # Strategy 1: Known classes
    potential_classes = ["div.av-purport", "div.wrapper-purport", "div.purport", "div.r-r-p"]
    
    text = ""
    
    for cls in potential_classes:
        t = safe_text(soup, cls)
        if t and len(t) > 10: 
            text = t
            break

    # Strategy 2: Find "Purport" text marker if class lookup failed
    if not text:
        marker = soup.find(string=lambda t: t and "Purport" in t.strip())
        if marker:
            parent = marker.find_parent("div")
            if parent:
                text = parent.get_text("\n\n", strip=True)

    # Always clean the "Purport" label if found
    return clean_label(text, "Purport")

def parse_verse_id(title_text):
    match = re.search(r"(\d+)\.(\d+)", title_text)
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


# ---------------- MAIN SCRAPER ---------------- #

all_verses = []
seen = set()

# Range is 1 to 18 (Python range stops before the end, so 1, 19)
for chapter in range(1, 19):
    print(f"\nScraping Chapter {chapter}")

    chapter_url = f"{BG_INDEX}{chapter}/"
    try:
        chapter_resp = requests.get(chapter_url, headers=HEADERS)
        if chapter_resp.status_code != 200:
            print(f"  Failed to load Chapter {chapter}")
            continue
            
        chapter_soup = BeautifulSoup(chapter_resp.text, "html.parser")

        verse_links = [
            a for a in chapter_soup.select("a[href]")
            if re.fullmatch(r"/en/library/bg/\d+/\d+/", a.get("href", ""))
        ]

        for link in verse_links:
            verse_url = BASE_URL + link["href"]
            
            try:
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

                # Extract and Clean specific fields
                translation_raw = safe_text(verse_soup, "div.av-translation")
                translation_clean = clean_label(translation_raw, "Translation")
                
                synonyms_raw = safe_text(verse_soup, "div.av-synonyms")
                synonyms_clean = clean_label(synonyms_raw, "Synonyms")

                purport_clean = extract_purport_robust(verse_soup)

                data = {
                    "verse_id": verse_id,
                    "chapter": ch,
                    "verse": vs,
                    "sanskrit": extract_sanskrit(verse_soup),
                    "synonyms": synonyms_clean,
                    "translation": translation_clean,
                    "purport": purport_clean,
                }

                all_verses.append(data)
                
                # Log status
                if purport_clean:
                    print(f"  ✔ {verse_id}")
                else:
                    print(f"  ⚠ {verse_id} (No purport found)")

                time.sleep(REQUEST_DELAY)
            
            except Exception as e:
                print(f"  ❌ Error scraping {verse_url}: {e}")

    except Exception as e:
        print(f"❌ Error loading chapter {chapter}: {e}")

# ---------------- SAVE ---------------- #

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(all_verses, f, ensure_ascii=False, indent=2)

print(f"\n DONE. Saved {len(all_verses)} verses to {OUTPUT_FILE}")