"""Extract text from knowledge-base PDFs into page-level JSONL records.

First stage of the vector RAG pipeline: PDFs in ``pdfs/`` are converted to
one JSONL file per document in ``processed/``, with one record per page.
Page-level records keep source/page provenance so downstream chunking,
metadata filtering, and citations can point back to the original document.

Image blocks are skipped, but a page that is entirely a scanned image (no text
layer) is rendered and OCR'd so its content is recovered rather than lost; those
pages are listed under ``ocr_pages`` in the manifest.

Non-content pages are filtered out before writing: table of contents, index,
glossary/contributor lists, reference lists (which may be interleaved per
chapter, not just at the end), preface/foreword, about-the-author, publisher
catalog pages, blank pages, and cover/copyright/dedication front matter.
Dropped pages are recorded per document in ``manifest.json`` for transparency.
When a reference or index list begins partway down a chapter's last page, the
chapter prose above it is salvaged and kept (those pages are listed under
``split_pages`` in the manifest).

Optionally (``--strip-headers``) each kept page's running header (book/chapter
title + page number) and recurring licensing/copyright footers are removed from
the emitted text so they do not pollute downstream chunks and embeddings; that
information is retained as chunk metadata.

Usage:
    python -m battery_aar.rag.scripts.process_pdfs [--pdf-dir DIR] [--out-dir DIR]
        [--strip-headers]
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np

RAG_DIR = Path(__file__).resolve().parent
DEFAULT_PDF_DIR = RAG_DIR / "pdfs"
DEFAULT_OUT_DIR = RAG_DIR / "processed"

# Pages averaging fewer characters than this are flagged as likely scanned.
LOW_TEXT_CHARS_PER_PAGE = 200

# Some books mix born-digital text with scanned image-only pages that carry no
# text layer. Such pages are rendered at this DPI and run through OCR so their
# content is not lost. OCR output shorter than this is treated as a figure-only
# page (no recoverable prose) and dropped as blank.
OCR_RENDER_DPI = 200
OCR_MIN_CHARS = 40

# Paragraphs shorter than this are dropped (figure axis labels, page numbers).
MIN_PARAGRAPH_CHARS = 4

# A paragraph repeated on at least this fraction of a document's pages (and on
# at least MIN_REPEAT_PAGES pages) is treated as a running header/footer or
# publisher watermark and removed.
REPEATED_PARAGRAPH_PAGE_FRACTION = 0.3
MIN_REPEAT_PAGES = 3

# Optional page-furniture stripping (--strip-headers). A page's first block is a
# running header ("Introduction 3", "16 2 Electrochemical Basics") when it is
# short and carries a standalone page-number token; longer lines are body prose.
HEADER_MAX_WORDS = 14
# Footers (bottom-of-page licensing / copyright / edition lines) recur across
# pages with only a page number or per-download UUID varying. They are removed
# when a page's last block(s) both recur (same normalized text on at least
# MIN_REPEAT_PAGES pages) AND carry one of these cues -- the cue gate keeps
# unique reference-list entries, which also cite publishers, from being caught.
FOOTER_SCAN_BLOCKS = 2  # only the last N blocks of a page are footer candidates
FOOTER_CUES = (
    "licensed to",
    "unauthenticated",
    "all rights reserved",
    "©",
    "c⃝",
    "springer imprint",
    "wiley-vch",
    "john wiley",
    "university of science and technology press",
    "first edition",
    "isbn",
)

# --- Non-content page classification -------------------------------------
# An index is a dense list of "term, page-number" entries. Some books punctuate
# it with commas (high comma density), others as "term 187, 322" (page-number
# heavy). Either the comma density OR the digit-token ratio confirms an index
# once the running header says "Index".
INDEX_COMMA_DENSITY = 15.0  # commas per 100 words
INDEX_DIGIT_RATIO = 0.25  # fraction of tokens containing a digit
# References carry many bracketed citation markers ("[12]") and commas.
REFERENCE_MIN_CITATIONS = 8  # "[N]" markers on the page
REFERENCE_COMMA_DENSITY = 12.0
# A reference/index list often begins partway down the last chapter page, below
# a bare "References"/"Index" heading. Chapter prose above that heading is kept
# as content if it is at least this long (below this it is a stray caption).
SPLIT_HEADING_WORDS = {
    "references": ("references", "bibliography"),
    "index": ("index",),
}
# A page qualifies as the start of the book body (ending the front matter) only
# if it has real prose: enough characters and few digit-heavy tokens.
MIN_CONTENT_CHARS = 200
CONTENT_MAX_DIGIT_RATIO = 0.25

# Running-header keywords (matched on the normalized first text block).
PREFACE_HEADERS = ("preface", "foreword", "acknowledg", "preamble")
GLOSSARY_HEADERS = (
    "glossary",
    "nomenclature",
    "abbreviations",
    "list of contributors",
    "list of figures",
    "list of tables",
    "list of symbols",
)
# Front-matter pages naming the authors or a companion website (kept out of the
# body). "authors" also covers a combined author-bio/copyright page.
FRONTMATTER_HEADERS = (
    "about the author",
    "authors",
    "about the companion",
)
# Publisher catalog / advertisement pages (e.g. "Recent Artech House Titles").
CATALOG_CUES = ("artech house", "listing of recent", "recent titles")
# Strong copyright cues used only to reject a front-matter page as the body
# anchor (see find_body_start). These phrases appear on copyright pages but not
# in body prose; publisher names and "©"/"C⃝" glyphs are deliberately excluded
# because they also appear as figure credits on real content pages.
COPYRIGHT_CUES = (
    "all rights reserved",
    "isbn",
    "first published",
    "library of congress",
)

_TEXT_BLOCK = 0  # fitz block type for text (1 = image)


@dataclass(frozen=True)
class PageRecord:
    doc_id: str
    source: str
    page: int  # 1-indexed
    n_pages: int
    n_chars: int
    text: str


def clean_text(raw: str) -> str:
    """Normalize extracted text without destroying scientific content."""
    # Fold ligatures and math-styled glyphs to plain forms ("deﬁne" -> "define",
    # "𝛼" -> "α") so keyword search matches what users actually type.
    text = unicodedata.normalize("NFKC", raw)
    # Rejoin words hyphenated across line breaks: "elec-\ntrode" -> "electrode".
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # Remaining single newlines inside a block are soft line wraps.
    text = text.replace("\n", " ")
    # Collapse runs of spaces/tabs.
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def extract_page_paragraphs(page: fitz.Page) -> list[str]:
    """Extract text from one page as a list of cleaned paragraph blocks."""
    blocks = page.get_text("blocks")
    paragraphs = [
        clean_text(block[4])
        for block in sorted(blocks, key=lambda b: (b[1], b[0]))
        if block[6] == _TEXT_BLOCK
    ]
    return [p for p in paragraphs if len(p) >= MIN_PARAGRAPH_CHARS]


_ocr_engine = None


def _get_ocr_engine():
    """Lazily construct the RapidOCR engine (loaded once, only if needed).

    RapidOCR is an optional dependency: books without scanned pages never
    trigger it, and its absence only matters when OCR is actually required."""
    global _ocr_engine
    if _ocr_engine is None:
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError(
                "scanned pages need OCR; install the RAG extras "
                "(pip install rapidocr-onnxruntime)"
            ) from exc
        _ocr_engine = RapidOCR()
    return _ocr_engine


def is_image_only_page(page: fitz.Page) -> bool:
    """A page with no text layer but at least one image (likely scanned)."""
    return not page.get_text().strip() and bool(page.get_images(full=True))


def ocr_page_paragraphs(page: fitz.Page) -> list[str]:
    """Render a scanned page and OCR it into cleaned paragraph blocks.

    Each recognized text line becomes one paragraph, matching the block-per-
    paragraph shape of :func:`extract_page_paragraphs` so downstream
    classification and boilerplate detection treat OCR'd pages the same way."""
    pixmap = page.get_pixmap(dpi=OCR_RENDER_DPI, colorspace=fitz.csGRAY)
    image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
        pixmap.height, pixmap.width, pixmap.n
    )
    result, _ = _get_ocr_engine()(image)
    if not result:
        return []
    paragraphs = [clean_text(line[1]) for line in result]
    return [p for p in paragraphs if len(p) >= MIN_PARAGRAPH_CHARS]


def find_boilerplate(pages: list[list[str]]) -> set[str]:
    """Paragraphs repeated across many pages: running headers, footers,
    publisher watermarks (e.g. Wiley download/license notices)."""
    page_counts: dict[str, int] = {}
    for paragraphs in pages:
        for paragraph in set(paragraphs):
            page_counts[paragraph] = page_counts.get(paragraph, 0) + 1
    threshold = max(MIN_REPEAT_PAGES, REPEATED_PARAGRAPH_PAGE_FRACTION * len(pages))
    return {p for p, count in page_counts.items() if count >= threshold}


_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I)


def _normalize_footer(block: str) -> str:
    """Fold the per-page variation out of a footer so copies match: strip the
    download UUID and digits (page numbers, years) that differ between pages."""
    text = _UUID.sub("", block.lower())
    text = re.sub(r"\d+", "#", text)
    return re.sub(r"\s+", " ", text).strip()


def find_footers(pages: list[list[str]]) -> set[str]:
    """Normalized bottom-of-page footers to strip: licensing / copyright /
    edition lines that recur across pages.

    A footer must both recur (same normalized last-block text on at least
    MIN_REPEAT_PAGES pages) and carry a licensing/copyright cue. The recurrence
    test excludes unique reference-list entries (which also name publishers),
    and the cue test excludes recurring body text (e.g. a repeated equation)."""
    counts: dict[str, int] = {}
    for paragraphs in pages:
        seen = {_normalize_footer(b) for b in paragraphs[-FOOTER_SCAN_BLOCKS:]}
        for norm in seen:
            counts[norm] = counts.get(norm, 0) + 1
    return {
        norm
        for norm, count in counts.items()
        if count >= MIN_REPEAT_PAGES and any(cue in norm for cue in FOOTER_CUES)
    }


def strip_footers(paragraphs: list[str], footers: set[str]) -> list[str]:
    """Drop trailing blocks whose normalized form is a known footer."""
    if not footers:
        return paragraphs
    kept = list(paragraphs)
    while kept and _normalize_footer(kept[-1]) in footers:
        kept.pop()
    return kept


def norm_header(paragraphs: list[str]) -> str:
    """The page's running header, lowercased and stripped of decorative glyphs
    and a leading/trailing page-number token.

    Only a *separate* number token is stripped, so the first letters of a real
    header survive ("232 References" -> "references", but "Index" stays "index",
    not "ndex"). Decorative running-header glyphs (e.g. Artech's chapter-title
    ornaments) collapse to spaces. If the first block is nothing but a page
    number ("XIII"), the next block is used as the header."""
    for block in paragraphs[:2]:
        header = block.lower().strip()
        header = re.sub(r"[^0-9a-z]+", " ", header).strip()  # glyphs/punct -> space
        header = re.sub(r"^(\d+|[ivxlcdm]+)\s+", "", header)  # "viii Contents"
        header = re.sub(r"\s+\d+$", "", header).strip()  # "References 233"
        if header and not re.fullmatch(r"[ivxlcdm\d]+", header):
            return header
    return ""


_PAGE_NUMBER_TOKEN = re.compile(r"^(\d+|[ivxlcdm]+)\s+|\s+(\d+|[ivxlcdm]+)$", re.I)


def strip_running_header(paragraphs: list[str]) -> list[str]:
    """Drop the leading running-header block (book/chapter title + page number).

    A running header is the page's first block when it is short and carries a
    standalone page-number token (leading "16 " or trailing " 21"), e.g.
    "Introduction 3" or "16 2 Electrochemical Basics". Genuine section openers
    ("Introduction", "Preface") have no page number and are kept, as is body
    prose (too long to match). Only the first block is considered, so at most one
    header is removed per page."""
    if not paragraphs:
        return paragraphs
    first = paragraphs[0].strip()
    if (
        len(first.split()) <= HEADER_MAX_WORDS
        and norm_header(paragraphs)  # has real title text beyond the page number
        and _PAGE_NUMBER_TOKEN.search(first)
    ):
        return paragraphs[1:]
    return paragraphs


def is_blank(text: str) -> bool:
    """A page with no extractable text (fully blank or image-only)."""
    return not text.strip()


def page_comma_density(text: str) -> float:
    """Commas per 100 whitespace tokens (words)."""
    n_words = len(text.split()) or 1
    return text.count(",") / n_words * 100


def digit_token_ratio(text: str) -> float:
    """Fraction of whitespace tokens containing at least one digit."""
    tokens = text.split()
    if not tokens:
        return 0.0
    return sum(1 for t in tokens if any(c.isdigit() for c in t)) / len(tokens)


def classify_page(paragraphs: list[str], text: str) -> str:
    """Label a page as content or a non-content section.

    ``paragraphs`` must be the boilerplate-filtered blocks so the header is the
    real running header, not a per-page publisher watermark. Header keywords are
    the primary signal; comma density and citation counts corroborate."""
    if is_blank(text):
        return "blank"

    header = norm_header(paragraphs)
    comma_density = page_comma_density(text)

    if header.startswith(PREFACE_HEADERS):
        return "preface"
    if "about the author" in header:
        return "about"
    if header.startswith(FRONTMATTER_HEADERS) or any(cue in header for cue in CATALOG_CUES):
        return "frontmatter"
    if header.startswith("contents") and comma_density < 5:
        return "toc"
    # An "Index" header confirms an index if it reads as an entry list: comma-
    # heavy, or (for page-number-style indices) digit-heavy.
    if "index" in header and (
        comma_density >= INDEX_COMMA_DENSITY or digit_token_ratio(text) >= INDEX_DIGIT_RATIO
    ):
        return "index"
    if header.startswith(("references", "bibliography")):
        return "references"
    citations = len(re.findall(r"\[\d+\]", text))
    if citations >= REFERENCE_MIN_CITATIONS and comma_density >= REFERENCE_COMMA_DENSITY:
        return "references"
    if header.startswith(GLOSSARY_HEADERS):
        return "glossary"
    return "content"


def salvage_leading_content(label: str, paragraphs: list[str]) -> str | None:
    """Chapter prose stranded above a mid-page reference/index list.

    When a chapter's last page ends and its reference (or index) list starts
    below a bare "References"/"Index" heading, the page is labeled ``references``
    or ``index`` and would be dropped whole. This returns the prose blocks above
    that heading (dropping the top-of-page running header, which repeats the same
    word), or ``None`` if the section starts at the top of the page or the
    leading prose is too short to be worth keeping."""
    words = SPLIT_HEADING_WORDS.get(label)
    if not words:
        return None
    heading = re.compile(r"(?:\d+\s+)?(?:%s)(?:\s+\d+)?" % "|".join(words))
    split = next(
        (i for i, block in enumerate(paragraphs) if block.strip().lower() in words),
        None,
    )
    if not split:  # None (no bare heading) or 0 (list starts at the top)
        return None
    lead = [b for b in paragraphs[:split] if not heading.fullmatch(b.strip().lower())]
    text = "\n\n".join(lead)
    return text if len(text) >= MIN_CONTENT_CHARS else None


def absorb_sandwiched_pages(labels: list[str]) -> list[str]:
    """Relabel a lone ``content`` page that sits between two pages of the same
    non-content section as that section.

    On books with alternating recto/verso running headers, a section's verso
    continuation pages carry the book-title header instead of the section header
    (e.g. a preface or index continuation), so they classify as ``content`` even
    though they belong to the surrounding section."""
    absorbed = list(labels)
    for i in range(1, len(labels) - 1):
        if labels[i] != "content":
            continue
        neighbor = labels[i - 1]
        if neighbor == labels[i + 1] and neighbor not in ("content", "blank"):
            absorbed[i] = neighbor
    return absorbed


def find_body_start(labels: list[str], pages: list[list[str]], texts: list[str]) -> int:
    """Index of the first page that begins the book body.

    A body anchor is a genuine prose page that is not a copyright/publisher page.
    Cover, title, "Related Titles", copyright, and dedication pages carry no
    reliable keyword, so everything still labeled ``content`` before this anchor
    is treated as front matter. (Preface/about/toc pages are already labeled by
    classify_page and skipped here.)"""
    for index, (label, paragraphs, text) in enumerate(zip(labels, pages, texts)):
        if label != "content":
            continue
        low = text.lower()
        if len(text) < 2500 and any(cue in low for cue in COPYRIGHT_CUES):
            continue  # copyright / publisher front matter, not the body
        if len(text) >= MIN_CONTENT_CHARS and digit_token_ratio(text) < CONTENT_MAX_DIGIT_RATIO:
            return index
    return 0


def process_pdf(
    pdf_path: Path, strip_headers: bool = False
) -> tuple[list[PageRecord], dict[str, list[int]], list[int], list[int]]:
    """Extract content pages from a PDF.

    Returns the kept content ``PageRecord``s, a map of dropped section label
    -> 1-indexed page numbers (toc, index, references, glossary, frontmatter),
    the page numbers whose chapter prose was salvaged from an otherwise dropped
    reference/index page, and the page numbers recovered via OCR.

    When ``strip_headers`` is set, each kept page's running header (book/chapter
    title + page number) and any recurring licensing/copyright footer are removed
    from the emitted text so they do not pollute downstream chunks and
    embeddings; that furniture is already captured as structured chunk metadata
    (title, page span). Stripping happens after classification, which relies on
    the header being present."""
    doc_id = pdf_path.stem
    ocr_pages: list[int] = []
    with fitz.open(pdf_path) as doc:
        n_pages = doc.page_count
        pages = []
        for index, page in enumerate(doc):
            paragraphs = extract_page_paragraphs(page)
            # Scanned pages have no text layer; recover their content with OCR.
            if not paragraphs and is_image_only_page(page):
                paragraphs = ocr_page_paragraphs(page)
                if sum(len(p) for p in paragraphs) >= OCR_MIN_CHARS:
                    ocr_pages.append(index + 1)
                else:
                    paragraphs = []  # figure-only page: nothing to keep
            pages.append(paragraphs)

    boilerplate = find_boilerplate(pages)
    kept_paragraphs = [[p for p in paragraphs if p not in boilerplate] for paragraphs in pages]
    # Recurring licensing/copyright footers vary per page (UUID, page number), so
    # find_boilerplate misses them; detect them by normalized recurrence + cue.
    footers = find_footers(kept_paragraphs) if strip_headers else set()
    texts = ["\n\n".join(paragraphs) for paragraphs in kept_paragraphs]

    labels = [classify_page(kept_paragraphs[i], texts[i]) for i in range(n_pages)]
    # Pull section continuation pages (whose header reverted to the book title)
    # back into their surrounding section before locating the body.
    labels = absorb_sandwiched_pages(labels)
    # Cover/title/copyright/dedication carry no keyword: everything still labeled
    # content before the body anchor is front matter.
    body_start = find_body_start(labels, kept_paragraphs, texts)
    for i in range(body_start):
        if labels[i] == "content":
            labels[i] = "frontmatter"

    records: list[PageRecord] = []
    dropped: dict[str, list[int]] = {}
    split_pages: list[int] = []
    for index, (label, text) in enumerate(zip(labels, texts)):
        page = index + 1
        if label != "content":
            # A reference/index page may still carry chapter prose above the
            # list; keep that prose as content and drop only the list below.
            salvaged = salvage_leading_content(label, kept_paragraphs[index])
            if salvaged is not None:
                text = salvaged
                split_pages.append(page)
            else:
                dropped.setdefault(label, []).append(page)
                continue
        elif strip_headers:
            # Drop the running header and recurring footer from a normal content
            # page. (Salvaged pages already exclude the header region and end
            # above the reference list, so only this branch needs it.)
            blocks = strip_footers(strip_running_header(kept_paragraphs[index]), footers)
            text = "\n\n".join(blocks)
        records.append(
            PageRecord(
                doc_id=doc_id,
                source=pdf_path.name,
                page=page,
                n_pages=n_pages,
                n_chars=len(text),
                text=text,
            )
        )
    return records, dropped, split_pages, ocr_pages


def write_jsonl(records: list[PageRecord], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def process_all(pdf_dir: Path, out_dir: Path, strip_headers: bool = False) -> dict:
    pdf_paths = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"no PDFs found in {pdf_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict = {
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "pdf_dir": str(pdf_dir),
        "strip_headers": strip_headers,
        "documents": [],
    }

    for pdf_path in pdf_paths:
        records, dropped, split_pages, ocr_pages = process_pdf(pdf_path, strip_headers=strip_headers)
        out_path = out_dir / f"{pdf_path.stem}.jsonl"
        write_jsonl(records, out_path)

        n_pages = records[0].n_pages if records else 0
        n_dropped = sum(len(pages) for pages in dropped.values())
        total_chars = sum(r.n_chars for r in records)
        low_text_pages = [r.page for r in records if r.n_chars < LOW_TEXT_CHARS_PER_PAGE]
        doc_summary = {
            "doc_id": pdf_path.stem,
            "source": pdf_path.name,
            "output": out_path.name,
            "n_pages": n_pages,
            "n_pages_kept": len(records),
            "n_chars": total_chars,
            "chars_per_page": round(total_chars / max(len(records), 1)),
            "low_text_pages": low_text_pages,
            "dropped_pages": {label: pages for label, pages in sorted(dropped.items())},
            "split_pages": split_pages,
            "ocr_pages": ocr_pages,
        }
        manifest["documents"].append(doc_summary)

        print(
            f"{pdf_path.name}: {len(records)}/{n_pages} pages kept, {total_chars} chars "
            f"-> {out_path.relative_to(out_dir.parent)}"
        )
        if n_dropped:
            summary = ", ".join(f"{label} {len(pages)}" for label, pages in sorted(dropped.items()))
            print(f"  dropped {n_dropped} non-content pages ({summary})")
        if split_pages:
            print(f"  salvaged chapter prose from {len(split_pages)} split pages: {split_pages}")
        if ocr_pages:
            print(f"  recovered {len(ocr_pages)} scanned pages via OCR: {ocr_pages}")
        if doc_summary["chars_per_page"] < LOW_TEXT_CHARS_PER_PAGE:
            print(
                f"  WARNING: {pdf_path.name} averages "
                f"{doc_summary['chars_per_page']} chars/page; likely scanned "
                "(text extraction will miss its content)"
            )
        elif low_text_pages:
            print(f"  note: low-text pages (possible figures/scans): {low_text_pages}")

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"manifest -> {manifest_path.relative_to(out_dir.parent)}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--strip-headers",
        action="store_true",
        help="remove running headers (book/chapter title + page number) and "
        "recurring licensing/copyright footers from page text; the same info is "
        "retained as chunk metadata",
    )
    args = parser.parse_args()
    process_all(args.pdf_dir, args.out_dir, strip_headers=args.strip_headers)


if __name__ == "__main__":
    main()