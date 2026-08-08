import io

import fitz
from docx import Document

from ai_lca.documents import extract_docx_text, extract_pdf_text


def test_pdf_extraction_keeps_page_markers():
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Nickel input: 2.5 kg")
    pdf_bytes = doc.tobytes()
    text = extract_pdf_text(pdf_bytes)
    assert "[PAGE 1]" in text
    assert "Nickel input" in text


def test_docx_extraction_keeps_paragraphs_and_tables_in_order():
    document = Document()
    document.add_heading("Methane pyrolysis", level=1)
    document.add_paragraph("The process produces hydrogen and solid carbon.")
    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Input"
    table.cell(0, 1).text = "Amount"
    table.cell(1, 0).text = "Electricity"
    table.cell(1, 1).text = "10 kWh/kg H2"

    buffer = io.BytesIO()
    document.save(buffer)

    text = extract_docx_text(buffer.getvalue())
    assert "[HEADING 1]" in text
    assert "Methane pyrolysis" in text
    assert "The process produces hydrogen" in text
    assert "[TABLE 1]" in text
    assert "Input | Amount" in text
    assert "Electricity | 10 kWh/kg H2" in text
    assert text.index("Methane pyrolysis") < text.index("[TABLE 1]")
