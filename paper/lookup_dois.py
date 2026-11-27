#!/usr/bin/env python3
"""
Enrich a .bib file with DOIs from Crossref (journals) and arXiv (preprints),
and add a https://doi.org/... URL to every entry that gets a DOI.

Usage:
    python lookup_dois.py main.bib --out main_with_dois.bib --report report.tsv
"""

import argparse
import re
import time
import unicodedata
import xml.etree.ElementTree as ET
from difflib import SequenceMatcher

import bibtexparser
import requests

MAILTO = "ammar.daskin@medeniyet.edu.tr"
CROSSREF_URL = "https://api.crossref.org/works"
ARXIV_API_URL = "https://export.arxiv.org/api/query"

HEADERS = {"User-Agent": f"bib-doi-lookup/1.0 (mailto:{MAILTO})"}

# arXiv DOI prefix (official) – see https://info.arxiv.org/help/doi.html
ARXIV_DOI_PREFIX = "10.48550/arXiv."


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def normalize(s: str) -> str:
    """Lowercase, strip accents, remove punctuation – for fuzzy matching."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^\w\s]", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize(a), normalize(b)).ratio()


def clean_latex(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r"\\[a-zA-Z]+\s*\{([^}]*)\}", r"\1", s)
    return s.replace("{", "").replace("}", "").strip()


def first_author_family(entry: dict) -> str:
    a = entry.get("author", "")
    if not a:
        return ""
    first = a.split(" and ")[0].strip()
    if "," in first:
        return first.split(",")[0].strip()
    return first.split()[-1] if first.split() else ""


def year_of(entry: dict) -> str:
    m = re.search(r"\d{4}", entry.get("year", ""))
    return m.group(0) if m else ""


def extract_arxiv_id(entry: dict) -> str:
    """
    Try to pull an arXiv identifier out of the entry.
    Looks in :eprint:, :journal:, :url:, and :note:.
    Returns e.g. "1411.4028", "quant-ph/0001106", "1809.01378".
    """
    candidates = []
    for field in ("eprint", "journal", "url", "note"):
        val = entry.get(field, "")
        if not val:
            continue
        candidates.append(val)

    # patterns that capture: 1411.4028, quant-ph/0001106, arXiv:1411.4028v2
    patterns = [
        r"arXiv\s*:\s*([a-z\-]+/\d{7}|\d{4}\.\d{4,5})",
        r"\b([a-z\-]+/\d{7})\b",
        r"\b(\d{4}\.\d{4,5})\b",
    ]
    for text in candidates:
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                return m.group(1).rstrip(".")
    return ""


# --------------------------------------------------------------------------
# Crossref
# --------------------------------------------------------------------------

def query_crossref(title: str, author_family: str, rows: int = 5):
    params = {
        "query.bibliographic": title,
        "rows": rows,
        "mailto": MAILTO,
        "select": "DOI,title,author,issued,container-title,type",
    }
    if author_family:
        params["query.author"] = author_family
    r = requests.get(CROSSREF_URL, params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()["message"]["items"]


def best_crossref_match(items, title: str, year: str):
    best, best_score = None, 0.0
    for it in items:
        t = it.get("title", [""])
        t = t[0] if t else ""
        score = similarity(title, t)
        if year:
            issued = it.get("issued", {}).get("date-parts", [[None]])[0][0]
            if str(issued) == year:
                score += 0.05
        if score > best_score:
            best_score, best = score, it
    return best, best_score


# --------------------------------------------------------------------------
# arXiv API
# --------------------------------------------------------------------------

def arxiv_lookup(arxiv_id: str):
    """
    Query the arXiv API for a single ID.
    Returns (arxiv_id, title) or (None, None) if not found.
    """
    params = {"id_list": arxiv_id}
    r = requests.get(ARXIV_API_URL, params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()
    root = ET.fromstring(r.text)

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    for entry in root.findall("atom:entry", ns):
        id_el = entry.find("atom:id", ns)
        title_el = entry.find("atom:title", ns)
        if id_el is None or title_el is None:
            continue
        # <id> is like http://arxiv.org/abs/1411.4028v1
        raw_id = id_el.text.strip().rsplit("/abs/", 1)[-1]
        # strip version suffix (v1, v2, …)
        raw_id = re.sub(r"v\d+$", "", raw_id)
        return raw_id, title_el.text.strip()
    return None, None


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bibfile")
    ap.add_argument("--out", help="write updated bib here (else not written)")
    ap.add_argument("--report", help="write TSV report here")
    ap.add_argument("--threshold", type=float, default=0.90,
                    help="min title similarity for Crossref (default 0.90)")
    ap.add_argument("--delay", type=float, default=1.0,
                    help="seconds between API calls (default 1.0)")
    args = ap.parse_args()

    parser = bibtexparser.bparser.BibTexParser(
        common_strings=True, ignore_nonstandard_types=False)
    with open(args.bibfile, encoding="utf-8") as f:
        bib = bibtexparser.load(f, parser=parser)

    rows = []
    added = 0

    for entry in bib.entries:
        key = entry["ID"]
        title = clean_latex(entry.get("title", ""))
        year = year_of(entry)
        author = first_author_family(entry)

        # ---- already has a DOI? just add/update URL ----
        if entry.get("doi"):
            entry["url"] = f"https://doi.org/{entry['doi']}"
            print(f"[skip]   {key}: already has DOI {entry['doi']}")
            rows.append([key, entry["doi"], "existing", "1.000", "doi"])
            continue

        doi, source = None, None

        # ---- 1) Crossref (journals / conferences) ----
        if title:
            try:
                items = query_crossref(title, author)
                best, score = best_crossref_match(items, title, year)
                if best is not None and score >= args.threshold:
                    doi = best["DOI"]
                    source = "crossref"
                    print(f"[ok]     {key}: {doi}   (Crossref, score {score:.3f})")
                    rows.append([key, doi, "matched", f"{score:.3f}", "crossref"])
                else:
                    print(f"[low]    {key}: no Crossref match (score {score:.3f})")
                    rows.append([key, "", "low", f"{score:.3f}", "crossref"])
            except Exception as e:
                print(f"[error]  {key}: Crossref – {e}")
                rows.append([key, "", f"error:{e}", "", "crossref"])
            time.sleep(args.delay)
        else:
            rows.append([key, "", "no_title", "", "crossref"])

        if doi:
            entry["doi"] = doi
            entry["url"] = f"https://doi.org/{doi}"
            added += 1
            continue

        # ---- 2) arXiv fallback (preprints) ----
        arxiv_id = extract_arxiv_id(entry)
        if arxiv_id:
            try:
                resolved_id, resolved_title = arxiv_lookup(arxiv_id)
                if resolved_id:
                    doi = ARXIV_DOI_PREFIX + resolved_id
                    entry["doi"] = doi
                    entry["url"] = f"https://doi.org/{doi}"
                    added += 1
                    print(f"[ok]     {key}: {doi}   (arXiv)")
                    rows.append([key, doi, "matched", "1.000", "arxiv"])
                else:
                    print(f"[low]    {key}: arXiv ID {arxiv_id} not found")
                    rows.append([key, "", f"arxiv_not_found:{arxiv_id}", "", "arxiv"])
            except Exception as e:
                print(f"[error]  {key}: arXiv – {e}")
                rows.append([key, "", f"error:{e}", "", "arxiv"])
            time.sleep(args.delay)
        else:
            print(f"[skip]   {key}: no arXiv ID found")
            rows.append([key, "", "no_arxiv_id", "", "arxiv"])

    # ---- write outputs ----
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            bibtexparser.dump(bib, f)
        print(f"\nWrote {args.out}  ({added} DOIs added)")

    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            f.write("key\tdoi\tstatus\tscore\tsource\n")
            for r in rows:
                f.write("\t".join(map(str, r)) + "\n")
        print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()

# pip install "bibtexparser<2" requests
# python lookup_dois.py main.bib --out main_with_dois.bib --report doi_report.tsv