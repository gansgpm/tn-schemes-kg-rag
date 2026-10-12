"""
Step 1 - Ingest.

tn.gov.in blocks automated scraping (robots.txt), so we work from pages you
save yourself in a browser:

  1. Open https://www.tn.gov.in/scheme_list.php?dep_id=Mg==
  2. Ctrl+S  ->  "Webpage, HTML only"  ->  save into data/raw/
  3. (Better) open each scheme's detail page from that list and save it too.
     Running this script prints every scheme link it finds, so you know
     which ones to open.

You can also drop .txt (copy-paste) or .pdf files into data/raw/.

Output: data/pages.json  -> [{source, title, text, links}]
"""
import json
import re
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from config import PAGES_JSON, RAW_DIR

BASE_URL = "https://www.tn.gov.in/"
NOISE_TAGS = ["script", "style", "noscript", "header", "footer", "nav", "form", "iframe"]
NOISE_PATTERNS = re.compile(r"(menu|footer|header|breadcrumb|sidebar|skip|social|banner)", re.I)


def clean_text(text: str) -> str:
    text = re.sub(r"[ \t\xa0]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def parse_html(path):
    """Strict cleaning first; if that strips the page bare, fall back to light cleaning."""
    html = path.read_text(encoding="utf-8", errors="ignore")
    page = _parse_html(html, path, strict=True)
    if len(page["text"]) < 300:
        light = _parse_html(html, path, strict=False)
        if len(light["text"]) > len(page["text"]):
            page = light
    if len(page["text"]) < 300:
        print(f"  [diagnose] {path.name}: file size {len(html):,} bytes, readable text {len(page['text'])} chars.")
        print("             The scheme list is probably loaded by JavaScript, so the saved HTML is an empty shell.")
        print("             Fix: open the page, Ctrl+A, Ctrl+C, paste into Notepad, save as data/raw/scheme_list.txt (UTF-8).")
    return page


def _parse_html(html, path, strict=True):
    soup = BeautifulSoup(html, "html.parser")
    title = (soup.title.get_text(strip=True) if soup.title else path.stem)

    # Collect scheme links before removing nav blocks
    links = []
    for a in soup.find_all("a", href=True):
        href, label = a["href"], a.get_text(" ", strip=True)
        if "scheme" in href.lower() and label and "scheme_list" not in href.lower():
            links.append({"text": label, "url": urljoin(BASE_URL, href)})

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    if strict:
        for tag in soup(["header", "footer", "nav", "form", "iframe"]):
            tag.decompose()
        for tag in soup.find_all(attrs={"class": NOISE_PATTERNS}) + soup.find_all(attrs={"id": NOISE_PATTERNS}):
            if not tag.decomposed:
                tag.decompose()

    # Keep table structure readable: "Label : Value" per row
    for tr in soup.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
        cells = [c for c in cells if c]
        tr.replace_with(soup.new_string("\n" + " : ".join(cells) + "\n"))

    main = (soup.find("main") or soup.find(id=re.compile("content", re.I)) or soup.body or soup) if strict else soup
    return {"source": path.name, "title": title,
            "text": clean_text(main.get_text("\n")), "links": links}


def parse_pdf(path):
    from pypdf import PdfReader
    text = "\n".join(p.extract_text() or "" for p in PdfReader(str(path)).pages)
    return {"source": path.name, "title": path.stem, "text": clean_text(text), "links": []}


def parse_txt(path):
    return {"source": path.name, "title": path.stem,
            "text": clean_text(path.read_text(encoding="utf-8", errors="ignore")), "links": []}


PARSERS = {".html": parse_html, ".htm": parse_html, ".pdf": parse_pdf, ".txt": parse_txt}


def main():
    files = sorted(p for p in RAW_DIR.glob("*") if p.suffix.lower() in PARSERS)
    if not files:
        raise SystemExit(f"No files in {RAW_DIR}. Save the scheme pages there first (see top of this file).")

    pages = []
    for f in files:
        page = PARSERS[f.suffix.lower()](f)
        if len(page["text"]) < 50:
            print(f"  skipped (almost empty): {f.name}")
            continue
        pages.append(page)
        print(f"  {f.name}: {len(page['text']):,} chars, {len(page['links'])} scheme links")

    PAGES_JSON.write_text(json.dumps(pages, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved {len(pages)} pages -> {PAGES_JSON}")

    all_links = {l["url"]: l["text"] for p in pages for l in p["links"]}
    if all_links:
        print("\nScheme detail pages found - save these into data/raw/ for richer answers:")
        for url, text in all_links.items():
            print(f"  - {text}: {url}")


if __name__ == "__main__":
    main()