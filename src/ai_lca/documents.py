from __future__ import annotations

import io
from pathlib import Path
from typing import Iterator

import fitz  # PyMuPDF
from docx import Document
from docx.document import Document as _Document
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph


def extract_pdf_text(pdf_bytes: bytes, max_pages: int | None = None) -> str:
    """Extract page-tagged text from a text-readable PDF locally."""
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


def _iter_docx_blocks(document: _Document) -> Iterator[Paragraph | Table]:
    """Yield DOCX paragraphs and tables in document order."""
    for child in document.element.body.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, document)
        elif isinstance(child, CT_Tbl):
            yield Table(child, document)


def extract_docx_text(docx_bytes: bytes) -> str:
    """Extract text from a DOCX while retaining paragraph/table provenance markers."""
    if not docx_bytes:
        raise ValueError("DOCX is empty")

    document = Document(io.BytesIO(docx_bytes))
    chunks: list[str] = []
    paragraph_no = 0
    table_no = 0

    for block in _iter_docx_blocks(document):
        if isinstance(block, Paragraph):
            text = block.text.strip()
            if text:
                paragraph_no += 1
                chunks.append(f"[PARAGRAPH {paragraph_no}]\n{text}")
        else:
            table_no += 1
            rows = []
            for row in block.rows:
                cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
                if any(cells):
                    rows.append("\t".join(cells))
            if rows:
                chunks.append(f"[TABLE {table_no}]\n" + "\n".join(rows))

    combined = "\n\n".join(chunks).strip()
    if not combined:
        raise ValueError("No readable text was found in the DOCX")
    return combined


def extract_document_text(file_bytes: bytes, filename: str) -> str:
    """Dispatch supported source documents to the appropriate local extractor."""
    suffix = Path(filename or "").suffix.lower()
    if suffix == ".pdf":
        return extract_pdf_text(file_bytes)
    if suffix == ".docx":
        return extract_docx_text(file_bytes)
    raise ValueError("Unsupported document type. Upload a PDF or DOCX file.")
