from ai_lca.benchmark_visual_evidence import ExpectedVisualRow, score_transcription


def test_score_transcription_reports_exact_missing_rows():
    rows = [
        ExpectedVisualRow("aec", "low-alloyed steel for container", "6075.6", "kg"),
        ExpectedVisualRow("aec", "concrete for foundation", "7.7", "m3"),
    ]
    text = "low-alloyed steel for container | 6075.6 | kg\n"

    report = score_transcription(text, rows)

    assert report["recall"] == 0.5
    assert report["matched_rows"] == 1
    assert report["missing_rows"] == 1
    assert report["missing"][0]["name"] == "concrete for foundation"


def test_score_transcription_accepts_numeric_format_variants():
    rows = [ExpectedVisualRow("pemec", "low-alloyed steel for container", "2250.0", "kg")]
    text = "low-alloyed steel for container | 2250 | kg\n"

    report = score_transcription(text, rows)

    assert report["recall"] == 1.0
