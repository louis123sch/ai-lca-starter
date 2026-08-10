from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator


CROSSREF_BASE = "https://api.crossref.org"
IJLCA_ONLINE_ISSN = "1614-7502"


@dataclass(frozen=True)
class LiteratureRecord:
    doi: str
    title: str
    published_year: int | None
    journal: str
    abstract: str | None
    url: str | None
    publisher: str | None
    type: str | None
    license_urls: tuple[str, ...]
    tdm_links: tuple[str, ...]

    @property
    def has_abstract(self) -> bool:
        return bool((self.abstract or "").strip())

    @property
    def has_tdm_link(self) -> bool:
        return bool(self.tdm_links)


def _get_json(url: str, *, timeout: float = 30.0, user_agent: str = "ai-lca-starter/0.4") -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": user_agent, "Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _first(values: object) -> str:
    if isinstance(values, list) and values:
        value = values[0]
        return str(value) if value is not None else ""
    return ""


def _year(item: dict) -> int | None:
    for key in ("published-print", "published-online", "published", "issued"):
        date_parts = (item.get(key) or {}).get("date-parts") or []
        if date_parts and date_parts[0]:
            try:
                return int(date_parts[0][0])
            except (TypeError, ValueError, IndexError):
                pass
    return None


def record_from_crossref(item: dict) -> LiteratureRecord:
    licenses = tuple(
        str(entry.get("URL"))
        for entry in (item.get("license") or [])
        if isinstance(entry, dict) and entry.get("URL")
    )
    tdm_links = tuple(
        str(entry.get("URL"))
        for entry in (item.get("link") or [])
        if isinstance(entry, dict) and entry.get("URL")
    )
    return LiteratureRecord(
        doi=str(item.get("DOI") or "").strip().lower(),
        title=_first(item.get("title")).strip(),
        published_year=_year(item),
        journal=_first(item.get("container-title")).strip(),
        abstract=(str(item.get("abstract")).strip() if item.get("abstract") else None),
        url=(str(item.get("URL")).strip() if item.get("URL") else None),
        publisher=(str(item.get("publisher")).strip() if item.get("publisher") else None),
        type=(str(item.get("type")).strip() if item.get("type") else None),
        license_urls=licenses,
        tdm_links=tdm_links,
    )


def iter_crossref_journal(
    issn: str,
    *,
    from_year: int | None = None,
    until_year: int | None = None,
    rows: int = 250,
    max_records: int | None = None,
    mailto: str | None = None,
    pause_seconds: float = 0.1,
) -> Iterator[LiteratureRecord]:
    """Stream journal metadata with Crossref cursor paging.

    This is intentionally metadata-only: Phase 1 discovery does not require PDFs.
    """
    if rows < 1 or rows > 1000:
        raise ValueError("rows must be between 1 and 1000")

    cursor = "*"
    emitted = 0
    while True:
        filters = ["type:journal-article"]
        if from_year is not None:
            filters.append(f"from-pub-date:{from_year}-01-01")
        if until_year is not None:
            filters.append(f"until-pub-date:{until_year}-12-31")

        params = {
            "filter": ",".join(filters),
            "rows": str(rows),
            "cursor": cursor,
            "select": "DOI,title,container-title,abstract,URL,publisher,type,license,link,published,published-online,published-print,issued",
        }
        if mailto:
            params["mailto"] = mailto
        url = f"{CROSSREF_BASE}/journals/{urllib.parse.quote(issn)}/works?{urllib.parse.urlencode(params)}"
        payload = _get_json(url)
        message = payload.get("message") or {}
        items = message.get("items") or []
        if not items:
            return

        for item in items:
            record = record_from_crossref(item)
            if not record.doi:
                continue
            yield record
            emitted += 1
            if max_records is not None and emitted >= max_records:
                return

        if len(items) < rows:
            return
        next_cursor = message.get("next-cursor")
        if not next_cursor or next_cursor == cursor:
            return
        cursor = str(next_cursor)
        if pause_seconds:
            time.sleep(pause_seconds)


def rough_lca_relevance(record: LiteratureRecord) -> int:
    """Cheap deterministic pre-screen before any LLM/full-text spend."""
    text = f"{record.title} {record.abstract or ''}".casefold()
    score = 0
    for term, weight in (
        ("life cycle assessment", 4),
        ("life-cycle assessment", 4),
        ("lca", 2),
        ("life cycle inventory", 3),
        ("inventory", 1),
        ("ecoinvent", 2),
        ("functional unit", 2),
        ("system boundary", 2),
        ("case study", 1),
    ):
        if term in text:
            score += weight
    for term, penalty in (
        ("editorial", 2),
        ("commentary", 2),
        ("correction", 4),
        ("erratum", 4),
    ):
        if term in text:
            score -= penalty
    return score


def write_catalogue(records: Iterable[LiteratureRecord], output_dir: Path) -> tuple[Path, Path, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "catalogue.jsonl"
    csv_path = output_dir / "catalogue.csv"
    rows: list[dict] = []
    for record in records:
        row = asdict(record)
        row["license_urls"] = list(record.license_urls)
        row["tdm_links"] = list(record.tdm_links)
        row["relevance_score"] = rough_lca_relevance(record)
        rows.append(row)

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    fieldnames = [
        "doi", "title", "published_year", "journal", "publisher", "type",
        "relevance_score", "abstract", "url", "license_urls", "tdm_links",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {key: row.get(key) for key in fieldnames}
            out["license_urls"] = " | ".join(row["license_urls"])
            out["tdm_links"] = " | ".join(row["tdm_links"])
            writer.writerow(out)
    return jsonl_path, csv_path, len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover large LCA literature corpora without downloading PDFs.")
    parser.add_argument("--issn", default=IJLCA_ONLINE_ISSN)
    parser.add_argument("--from-year", type=int)
    parser.add_argument("--until-year", type=int)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--rows", type=int, default=250)
    parser.add_argument("--mailto")
    parser.add_argument("--output-dir", type=Path, default=Path("literature_runs/ijlca"))
    args = parser.parse_args()

    records = iter_crossref_journal(
        args.issn,
        from_year=args.from_year,
        until_year=args.until_year,
        rows=args.rows,
        max_records=args.max_records,
        mailto=args.mailto,
    )
    jsonl_path, csv_path, count = write_catalogue(records, args.output_dir)
    print(f"Discovered {count} records")
    print(f"JSONL: {jsonl_path}")
    print(f"CSV:   {csv_path}")


if __name__ == "__main__":
    main()
