from __future__ import annotations

import io

import fitz  # PyMuPDF
from docx import Document
from docx.document import Document as DocxDocument
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph


def extract_pdf_text(pdf_bytes: bytes, max_pages: int | None = None) -> str:
    """Extract page-tagged text from a text-readable PDF locally.

    Page markers are deliberately retained so the LLM can return page provenance.
    This prototype does not OCR scanned/image-only PDFs.
    """
    if not pdf_bytes:
        raise ValueError("PDF is empty")

    doc = fitz.open(stream=io.BytesIO(pdf_bytes), filetype="pdf")
    chunks: list[str] = []
    n_pages = len(doc) if max_pages is None else min(len(doc), max_pages)

    for idx in range(n_pages):
        page = doc[idx]
        text = page.get_text("text").strip()
        chunks.append(f"[PAGE {idx + 1}]\n{text}")

    combined = "\n\n".join(chunks).strip()
    if not combined:
        raise ValueError(
            "No machine-readable text was found. This PDF may be scanned/image-only; "
            "use pasted text for now or add multimodal/OCR ingestion later."
        )
    return combined


def _iter_docx_blocks(document: DocxDocument):
    """Yield Word paragraphs and tables in their original document order."""
    for child in document.element.body.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, document)
        elif isinstance(child, CT_Tbl):
            yield Table(child, document)


def extract_docx_text(docx_bytes: bytes) -> str:
    """Extract ordered text and tables from a modern Word (.docx) document.

    DOCX files do not contain reliable fixed page boundaries, so page provenance is
    deliberately left unavailable. Tables are retained with explicit [TABLE N]
    markers because LCI values are frequently presented in tables.
    """
    if not docx_bytes:
        raise ValueError("Word document is empty")

    try:
        document = Document(io.BytesIO(docx_bytes))
    except Exception as exc:
        raise ValueError(f"Could not read Word document: {exc}") from exc

    chunks: list[str] = []
    table_number = 0

    for block in _iter_docx_blocks(document):
        if isinstance(block, Paragraph):
            text = block.text.strip()
            if not text:
                continue
            style_name = (block.style.name or "").strip() if block.style else ""
            if style_name.lower().startswith("heading"):
                chunks.append(f"[{style_name.upper()}]\n{text}")
            else:
                chunks.append(text)
            continue

        table_number += 1
        rows: list[str] = []
        for row in block.rows:
            cells = [" ".join(cell.text.split()) for cell in row.cells]
            if any(cells):
                rows.append(" | ".join(cells))
        if rows:
            chunks.append(f"[TABLE {table_number}]\n" + "\n".join(rows))

    combined = "\n\n".join(chunks).strip()
    if not combined:
        raise ValueError("No readable text or table content was found in the Word document.")
    return combined
