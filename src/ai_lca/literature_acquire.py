from __future__ import annotations

import argparse
import json
import os
import urllib.parse
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path


SPRINGER_API_BASE = "https://api.springernature.com"


@dataclass(frozen=True)
class AcquisitionResult:
    doi: str
    status: str
    source_type: str | None
    path: str | None
    detail: str | None = None


def springer_query_url(doi: str, api_key: str, *, tdm: bool = False) -> str:
    endpoint = "/xmldata/jats" if tdm else "/openaccess/jats"
    params = urllib.parse.urlencode({"q": f"doi:{doi}", "api_key": api_key})
    return f"{SPRINGER_API_BASE}{endpoint}?{params}"


def _fetch_bytes(url: str, *, timeout: float = 60.0) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "ai-lca-starter/0.4",
            "Accept": "application/xml,text/xml,application/jats+xml,*/*;q=0.1",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def acquire_springer_jats(
    doi: str,
    api_key: str,
    output_dir: Path,
    *,
    allow_tdm: bool = False,
) -> AcquisitionResult:
    """Acquire machine-readable Springer Nature content without using a PDF.

    Open-access JATS is attempted first. TDM full text is attempted only when the
    caller explicitly enables it, because access depends on the applicable agreement.
    """
    safe = doi.lower().replace("/", "__")
    paper_dir = output_dir / safe
    paper_dir.mkdir(parents=True, exist_ok=True)

    attempts = [("springer_oa_jats", False)]
    if allow_tdm:
        attempts.append(("springer_tdm_jats", True))

    errors: list[str] = []
    for source_type, tdm in attempts:
        try:
            content = _fetch_bytes(springer_query_url(doi, api_key, tdm=tdm))
        except Exception as exc:  # network/provider failures become queue statuses
            errors.append(f"{source_type}: {type(exc).__name__}: {exc}")
            continue
        stripped = content.lstrip()
        if not stripped.startswith(b"<"):
            errors.append(f"{source_type}: response was not XML")
            continue
        path = paper_dir / f"{source_type}.xml"
        path.write_bytes(content)
        return AcquisitionResult(doi=doi, status="acquired", source_type=source_type, path=str(path))

    return AcquisitionResult(
        doi=doi,
        status="unavailable_automatically",
        source_type=None,
        path=None,
        detail=" | ".join(errors) if errors else "No machine-readable source was returned.",
    )


def iter_catalogue_dois(path: Path, *, min_relevance: int = 0):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if int(row.get("relevance_score", 0)) < min_relevance:
                continue
            doi = str(row.get("doi") or "").strip()
            if doi:
                yield doi


def main() -> None:
    parser = argparse.ArgumentParser(description="Acquire machine-readable LCA sources without PDFs.")
    parser.add_argument("catalogue", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("literature_runs/sources"))
    parser.add_argument("--min-relevance", type=int, default=4)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--allow-tdm", action="store_true")
    args = parser.parse_args()

    api_key = os.environ.get("SPRINGER_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("SPRINGER_API_KEY is required for Springer Nature machine-readable acquisition.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.output_dir / "acquisition_status.jsonl"
    count = 0
    with status_path.open("w", encoding="utf-8") as status:
        for doi in iter_catalogue_dois(args.catalogue, min_relevance=args.min_relevance):
            result = acquire_springer_jats(doi, api_key, args.output_dir, allow_tdm=args.allow_tdm)
            status.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
            count += 1
            print(result.status, doi, result.source_type or "")
            if args.max_records is not None and count >= args.max_records:
                break
    print(f"Processed {count} DOI(s); status log: {status_path}")


if __name__ == "__main__":
    main()
