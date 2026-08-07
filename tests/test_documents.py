import fitz
from ai_lca.documents import extract_pdf_text


def test_pdf_extraction_keeps_page_markers():
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Nickel input: 2.5 kg")
    pdf_bytes = doc.tobytes()
    text = extract_pdf_text(pdf_bytes)
    assert "[PAGE 1]" in text
    assert "Nickel input" in text
