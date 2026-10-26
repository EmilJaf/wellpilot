"""Render Volve report PDFs into page images and text (no LLM involved).

Each page becomes a PNG that Muse Spark can read, plus its text layer when the PDF has
one. Text is normalized so quotes can later be checked against it.

Run: uv run python -m wellpilot.ingest.reports
"""

import argparse
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pypdfium2 as pdfium

from wellpilot.config import get_settings

PAGES_FILE = "report_pages.parquet"
RENDER_WIDTH_PX = 1240  # A4 at 150 dpi: about 1.4k image tokens per page for Muse Spark


@dataclass(frozen=True)
class Report:
    doc_id: str
    title: str
    file: str  # under data/raw/reports
    source: str  # path in the Volve dataset
    language: str
    doc_date: str  # ISO date the document was issued, for as-of filtering


REPORTS = [
    Report(
        doc_id="volve_pud",
        title="Plan for utbygging og drift av Volve (PUD, plan for development and operation)",
        file="Volve_PUD.pdf",
        source="Reports/Volve PUD .pdf",
        language="no",
        doc_date="2005-02-01",
    ),
]
REPORTS_BY_ID = {r.doc_id: r for r in REPORTS}


def normalize_text(text: str) -> str:
    """Join soft-hyphenated words and collapse whitespace.

    pdfium marks soft hyphens as U+FFFE (or U+00AD); words broken at a line end with a
    plain hyphen are joined too, so a quote matches however the line was wrapped.
    """
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[￾­]\s*", "", text)
    text = re.sub(r"(?<=[a-zæøå])-\s*\n\s*(?=[a-zæøå])", "", text)
    return " ".join(text.split())


def page_dir(processed_dir: Path, doc_id: str) -> Path:
    return processed_dir / "reports" / doc_id


def render_report(report: Report, raw_dir: Path, processed_dir: Path) -> pd.DataFrame:
    pdf = pdfium.PdfDocument(raw_dir / "reports" / report.file)
    out = page_dir(processed_dir, report.doc_id)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in range(len(pdf)):
        page = pdf[index]
        width_pt, _ = page.get_size()
        image = page.render(scale=RENDER_WIDTH_PX / width_pt).to_pil()
        name = f"p{index + 1:03d}.png"
        image.save(out / name, optimize=True)
        text = normalize_text(page.get_textpage().get_text_range())
        rows.append(
            {
                "doc_id": report.doc_id,
                "page": index + 1,
                "pages": len(pdf),
                "text": text,
                "has_text": len(text) >= 50,
                "image": f"reports/{report.doc_id}/{name}",
                "width_px": image.width,
                "height_px": image.height,
            }
        )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Render report PDFs to page PNGs and text.")
    parser.add_argument("--doc", choices=sorted(REPORTS_BY_ID), action="append")
    args = parser.parse_args(argv)

    frames = []
    for doc_id in args.doc or sorted(REPORTS_BY_ID):
        pages = render_report(REPORTS_BY_ID[doc_id], settings.raw_dir, settings.processed_dir)
        print(f"{doc_id}: {len(pages)} pages, {int(pages['has_text'].sum())} with text")
        frames.append(pages)

    out = settings.processed_dir / PAGES_FILE
    if out.exists():  # keep pages of documents not rendered this time
        old = pd.read_parquet(out)
        frames.insert(0, old[~old["doc_id"].isin({f["doc_id"].iloc[0] for f in frames})])
    pd.concat(frames, ignore_index=True).to_parquet(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
