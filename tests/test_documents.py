import io

import fitz
from docx import Document

from ai_lca.documents import extract_document_text, extract_docx_text, extract_pdf_text


def test_pdf_extraction_keeps_page_markers():
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Nickel input: 2.5 kg")
    pdf_bytes = doc.tobytes()
    text = extract_pdf_text(pdf_bytes)
    assert "[PAGE 1]" in text
    assert "Nickel input" in text


def test_docx_extraction_keeps_paragraph_and_table_markers():
    document = Document()
    document.add_paragraph("Natural gas input: 1.2 kg")
    table = document.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "Electricity"
    table.cell(0, 1).text = "5 kWh"
    buffer = io.BytesIO()
    document.save(buffer)

    text = extract_docx_text(buffer.getvalue())
    assert "[PARAGRAPH 1]" in text
    assert "Natural gas input" in text
    assert "[TABLE 1]" in text
    assert "Electricity\t5 kWh" in text


def test_document_dispatch_accepts_docx():
    document = Document()
    document.add_paragraph("Hydrogen production")
    buffer = io.BytesIO()
    document.save(buffer)
    assert "Hydrogen production" in extract_document_text(buffer.getvalue(), "paper.docx")


def test_document_dispatch_accepts_utf8_text_fixture():
    text = "[PAGE 11]\nLCI data of SMR-based hydrogen production"
    assert extract_document_text(text.encode("utf-8"), "paper_fixture.txt") == text


def test_combined_documents_keep_document_markers():
    from ai_lca.documents import combine_document_texts

    document = Document()
    document.add_paragraph("Supplementary inventory")
    buffer = io.BytesIO()
    document.save(buffer)

    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "Main paper inventory")
    pdf_bytes = pdf.tobytes()

    text = combine_document_texts([
        ("paper.pdf", pdf_bytes),
        ("supplement.docx", buffer.getvalue()),
    ])
    assert "[DOCUMENT: paper.pdf]" in text
    assert "[DOCUMENT: supplement.docx]" in text
    assert "Main paper inventory" in text
    assert "Supplementary inventory" in text
