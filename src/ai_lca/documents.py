from __future__ import annotations

import io
import fitz  # PyMuPDF


def extract_pdf_text(pdf_bytes: bytes, max_pages: int | None = None) -> str:
    """Extract page-tagged text from a text-readable PDF locally.

    Page markers are deliberately retained so the LLM can return page provenance.
    This first prototype does not OCR scanned/image-only PDFs.
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
