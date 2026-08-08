from ai_lca.documents import extract_pdf_text


def test_extract_pdf_text_rejects_empty_bytes():
    try:
        extract_pdf_text(b"")
    except ValueError as exc:
        assert "empty" in str(exc).lower()
    else:
        raise AssertionError("Expected empty PDF bytes to raise ValueError")
