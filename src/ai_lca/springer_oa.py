from __future__ import annotations

import argparse
import csv
import os
import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET


OA_JATS_ENDPOINT = "https://api.springernature.com/openaccess/jats"


@dataclass(frozen=True)
class AcquisitionResult:
    doi: str
    status: str
    path: str | None
    bytes_written: int
    detail: str | None = None


def _slug_doi(doi: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", doi.strip().lower())


def _has_article_content(xml_bytes: bytes) -> bool:
    """Return True only when the OA response contains an article/book record."""
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return False
    for elem in root.iter():
        tag = elem.tag.rsplit("}", 1)[-1]
        if tag in {"article", "book-part-wrapper"}:
            return True
    return False


def fetch_jats_for_doi(
    doi: str,
    api_key: str,
    *,
    timeout: float = 45.0,
    user_agent: str = "ai-lca-starter/0.4",
) -> bytes | None:
    params = urllib.parse.urlencode({"q": f"doi:{doi}", "api_key": api_key})
    request = urllib.request.Request(
        f"{OA_JATS_ENDPOINT}?{params}",
        headers={"User-Agent": user_agent, "Accept": "application/xml,text/xml;q=0.9,*/*;q=0.1"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = response.read()
    return data if _has_article_content(data) else None


def acquire_catalogue(
    catalogue_csv: Path,
    output_dir: Path,
    *,
    api_key: str,
    max_attempts: int | None = None,
    min_relevance: int | None = None,
    pause_seconds: float = 0.2,
) -> list[AcquisitionResult]:
    rows = list(csv.DictReader(catalogue_csv.open(encoding="utf-8")))
    rows.sort(key=lambda row: int(row.get("relevance_score") or 0), reverse=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[AcquisitionResult] = []
    attempts = 0

    for row in rows:
        doi = (row.get("doi") or "").strip().lower()
        if not doi:
            continue
        score = int(row.get("relevance_score") or 0)
        if min_relevance is not None and score < min_relevance:
            continue
        if max_attempts is not None and attempts >= max_attempts:
            break
        attempts += 1
        try:
            data = fetch_jats_for_doi(doi, api_key)
            if data is None:
                results.append(AcquisitionResult(doi, "not_open_access_jats", None, 0))
            else:
                path = output_dir / f"{_slug_doi(doi)}.xml"
                path.write_bytes(data)
                results.append(AcquisitionResult(doi, "acquired", str(path), len(data)))
        except Exception as exc:  # keep large batches moving; failures are logged per DOI
            results.append(AcquisitionResult(doi, "error", None, 0, f"{type(exc).__name__}: {exc}"))
        if pause_seconds:
            time.sleep(pause_seconds)

    manifest = output_dir / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["doi", "status", "path", "bytes_written", "detail"])
        writer.writeheader()
        for item in results:
            writer.writerow({
                "doi": item.doi,
                "status": item.status,
                "path": item.path or "",
                "bytes_written": item.bytes_written,
                "detail": item.detail or "",
            })
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Acquire machine-readable Springer OA JATS by DOI; never downloads PDFs.")
    parser.add_argument("catalogue", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("literature_runs/ijlca/jats"))
    parser.add_argument("--max-attempts", type=int)
    parser.add_argument("--min-relevance", type=int)
    parser.add_argument("--api-key-env", default="SPRINGER_API_KEY")
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise SystemExit(f"Missing required environment variable: {args.api_key_env}")

    results = acquire_catalogue(
        args.catalogue,
        args.output_dir,
        api_key=api_key,
        max_attempts=args.max_attempts,
        min_relevance=args.min_relevance,
    )
    acquired = sum(item.status == "acquired" for item in results)
    unavailable = sum(item.status == "not_open_access_jats" for item in results)
    errors = sum(item.status == "error" for item in results)
    print(f"attempted={len(results)} acquired={acquired} unavailable={unavailable} errors={errors}")
    print(f"manifest={args.output_dir / 'manifest.csv'}")


if __name__ == "__main__":
    main()
