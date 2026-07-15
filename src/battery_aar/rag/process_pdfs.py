"""Extract text from knowledge-base PDFs into page-level JSONL records.

First stage of the vector RAG pipeline: PDFs in ``pdfs/`` are converted to
one JSONL file per document in ``processed/``, with one record per page.
Page-level records keep source/page provenance so downstream chunking,
metadata filtering, and citations can point back to the original document.

Text-only by design: image blocks are skipped and pages that yield almost
no text (likely scanned) are reported so they are not silently lost.

Usage:
    python -m battery_aar.rag.process_pdfs [--pdf-dir DIR] [--out-dir DIR]
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

RAG_DIR = Path(__file__).resolve().parent
DEFAULT_PDF_DIR = RAG_DIR / "pdfs"
DEFAULT_OUT_DIR = RAG_DIR / "processed"

# Pages averaging fewer characters than this are flagged as likely scanned.
LOW_TEXT_CHARS_PER_PAGE = 200

# Paragraphs shorter than this are dropped (figure axis labels, page numbers).
MIN_PARAGRAPH_CHARS = 4

# A paragraph repeated on at least this fraction of a document's pages (and on
# at least MIN_REPEAT_PAGES pages) is treated as a running header/footer or
# publisher watermark and removed.
REPEATED_PARAGRAPH_PAGE_FRACTION = 0.3
MIN_REPEAT_PAGES = 3

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


def find_boilerplate(pages: list[list[str]]) -> set[str]:
    """Paragraphs repeated across many pages: running headers, footers,
    publisher watermarks (e.g. Wiley download/license notices)."""
    page_counts: dict[str, int] = {}
    for paragraphs in pages:
        for paragraph in set(paragraphs):
            page_counts[paragraph] = page_counts.get(paragraph, 0) + 1
    threshold = max(MIN_REPEAT_PAGES, REPEATED_PARAGRAPH_PAGE_FRACTION * len(pages))
    return {p for p, count in page_counts.items() if count >= threshold}


def process_pdf(pdf_path: Path) -> list[PageRecord]:
    doc_id = pdf_path.stem
    with fitz.open(pdf_path) as doc:
        n_pages = doc.page_count
        pages = [extract_page_paragraphs(page) for page in doc]

    boilerplate = find_boilerplate(pages)
    records = []
    for index, paragraphs in enumerate(pages):
        text = "\n\n".join(p for p in paragraphs if p not in boilerplate)
        records.append(
            PageRecord(
                doc_id=doc_id,
                source=pdf_path.name,
                page=index + 1,
                n_pages=n_pages,
                n_chars=len(text),
                text=text,
            )
        )
    return records


def write_jsonl(records: list[PageRecord], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def process_all(pdf_dir: Path, out_dir: Path) -> dict:
    pdf_paths = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"no PDFs found in {pdf_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict = {
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "pdf_dir": str(pdf_dir),
        "documents": [],
    }

    for pdf_path in pdf_paths:
        records = process_pdf(pdf_path)
        out_path = out_dir / f"{pdf_path.stem}.jsonl"
        write_jsonl(records, out_path)

        total_chars = sum(r.n_chars for r in records)
        low_text_pages = [r.page for r in records if r.n_chars < LOW_TEXT_CHARS_PER_PAGE]
        doc_summary = {
            "doc_id": pdf_path.stem,
            "source": pdf_path.name,
            "output": out_path.name,
            "n_pages": len(records),
            "n_chars": total_chars,
            "chars_per_page": round(total_chars / max(len(records), 1)),
            "low_text_pages": low_text_pages,
        }
        manifest["documents"].append(doc_summary)

        print(
            f"{pdf_path.name}: {len(records)} pages, {total_chars} chars "
            f"-> {out_path.relative_to(out_dir.parent)}"
        )
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
    args = parser.parse_args()
    process_all(args.pdf_dir, args.out_dir)


if __name__ == "__main__":
    main()
