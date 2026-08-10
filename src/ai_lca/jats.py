from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree as ET

_NUMERIC_RE = re.compile(r"(?<![A-Za-z])[-+]?(?:\d+(?:[.,]\d+)?|\.\d+)(?:\s*[×x]\s*10\s*[-^]?\s*\d+)?")
_UNIT_RE = re.compile(
    r"\b(?:kg|g|mg|µg|ug|t|tonnes?|kwh|mwh|wh|mj|gj|kj|j|l|ml|m3|m²|m2|cm²|cm2|km|m|cm|mm|tkm|mol|mmol|%|pcs?|pieces?|units?)\b",
    re.IGNORECASE,
)
_LCI_SECTION_RE = re.compile(
    r"(?:life.?cycle|inventory|lci|materials?|methods?|goal|scope|system.?bound|foreground|"
    r"manufactur|production|construction|operation|use.?phase|end.?of.?life|recycl|input|output)",
    re.IGNORECASE,
)

def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]

def _clean_text(element: ET.Element | None) -> str:
    if element is None:
        return ""
    return " ".join("".join(element.itertext()).split())

def _stable_candidate_id(source_location: str, text: str) -> str:
    digest = hashlib.sha1(f"{source_location}\n{text}".encode("utf-8")).hexdigest()[:10]
    return f"cand_{digest}"

@dataclass(frozen=True)
class InventoryCandidate:
    candidate_id: str
    source_location: str
    evidence_text: str
    context: str
    evidence_type: str
    table: str | None = None

    def as_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "source_location": self.source_location,
            "evidence_text": self.evidence_text,
            "context": self.context,
            "evidence_type": self.evidence_type,
            "table": self.table,
        }

@dataclass
class JATSDocument:
    doi: str | None
    title: str
    abstract: str
    sections: list[tuple[str, str]]
    tables: list[tuple[str, str, list[str]]]
    inventory_candidates: list[InventoryCandidate] = field(default_factory=list)

    @property
    def section_titles(self) -> list[str]:
        return [title for title, _ in self.sections]

    def screening_text(self, max_chars: int = 18000) -> str:
        parts = [
            f"TITLE: {self.title}",
            f"DOI: {self.doi or 'not identified'}",
            f"ABSTRACT: {self.abstract}",
            "SECTION TITLES: " + " | ".join(self.section_titles[:80]),
            f"TABLE COUNT: {len(self.tables)}",
            f"DETERMINISTIC INVENTORY CANDIDATE COUNT: {len(self.inventory_candidates)}",
        ]
        for label, caption, rows in self.tables[:12]:
            parts.append(f"[TABLE {label}] {caption}\n" + "\n".join(rows[:8]))
        return "\n\n".join(parts)[:max_chars]

    def structure_text(self, max_chars: int = 120000) -> str:
        parts = [
            f"[DOCUMENT: JATS XML]\n[TITLE]\n{self.title}",
            f"[ABSTRACT]\n{self.abstract}",
        ]
        for title, text in self.sections:
            if _LCI_SECTION_RE.search(title) or _LCI_SECTION_RE.search(text[:800]):
                parts.append(f"[SECTION: {title or 'untitled'}]\n{text}")
        for label, caption, rows in self.tables:
            row_text = "\n".join(f"[TABLE: {label} | ROW {idx}] {row}" for idx, row in enumerate(rows, 1))
            parts.append(f"[TABLE: {label}]\nCAPTION: {caption}\n{row_text}")
        return "\n\n".join(parts)[:max_chars]

def _find_article(root: ET.Element, expected_doi: str | None = None) -> ET.Element:
    articles = [root] if _local(root.tag) == "article" else [e for e in root.iter() if _local(e.tag) == "article"]
    if not articles:
        raise ValueError("No JATS <article> element found")
    if expected_doi:
        target = expected_doi.strip().lower()
        for article in articles:
            for elem in article.iter():
                if _local(elem.tag) == "article-id" and (elem.attrib.get("pub-id-type") or "").lower() == "doi":
                    if _clean_text(elem).lower() == target:
                        return article
    return articles[0]

def _article_doi(article: ET.Element) -> str | None:
    for elem in article.iter():
        if _local(elem.tag) == "article-id" and (elem.attrib.get("pub-id-type") or "").lower() == "doi":
            value = _clean_text(elem)
            return value.lower() if value else None
    return None

def _section_content(sec: ET.Element) -> tuple[str, str]:
    title_elem = next((c for c in list(sec) if _local(c.tag) == "title"), None)
    title = _clean_text(title_elem)
    chunks: list[str] = []
    for child in list(sec):
        name = _local(child.tag)
        if name == "p":
            txt = _clean_text(child)
            if txt:
                chunks.append(txt)
        elif name == "list":
            for item in child.iter():
                if _local(item.tag) == "list-item":
                    txt = _clean_text(item)
                    if txt:
                        chunks.append(f"- {txt}")
    return title, "\n".join(chunks)

def parse_jats_bytes(xml_bytes: bytes, *, expected_doi: str | None = None) -> JATSDocument:
    root = ET.fromstring(xml_bytes)
    article = _find_article(root, expected_doi)
    title_elem = next((e for e in article.iter() if _local(e.tag) == "article-title"), None)
    title = _clean_text(title_elem)
    abstract_elem = next((e for e in article.iter() if _local(e.tag) == "abstract"), None)
    abstract = _clean_text(abstract_elem)

    sections: list[tuple[str, str]] = []
    for sec in article.iter():
        if _local(sec.tag) != "sec":
            continue
        section_title, section_text = _section_content(sec)
        if section_title or section_text:
            sections.append((section_title, section_text))

    tables: list[tuple[str, str, list[str]]] = []
    candidates: list[InventoryCandidate] = []
    for table_index, wrap in enumerate((e for e in article.iter() if _local(e.tag) == "table-wrap"), 1):
        label_elem = next((c for c in list(wrap) if _local(c.tag) == "label"), None)
        label = _clean_text(label_elem) or f"T{table_index}"
        caption_elem = next((c for c in list(wrap) if _local(c.tag) == "caption"), None)
        caption = _clean_text(caption_elem)
        rows: list[str] = []
        header_rows: list[str] = []
        for row_index, tr in enumerate((e for e in wrap.iter() if _local(e.tag) == "tr"), 1):
            cells = [c for c in list(tr) if _local(c.tag) in {"td", "th"}]
            cell_texts = [_clean_text(c) for c in cells]
            cell_texts = [t for t in cell_texts if t]
            if not cell_texts:
                continue
            row_text = " | ".join(cell_texts)
            rows.append(row_text)
            if any(_local(c.tag) == "th" for c in cells):
                header_rows.append(row_text)
            if len(cell_texts) >= 2 and (_NUMERIC_RE.search(row_text) or _UNIT_RE.search(row_text)):
                location = f"table:{label}:row:{row_index}"
                context = f"caption={caption or 'none'}"
                if header_rows:
                    context += f"; headers={' || '.join(header_rows[-3:])}"
                candidates.append(InventoryCandidate(
                    candidate_id=_stable_candidate_id(location, row_text),
                    source_location=location,
                    evidence_text=row_text,
                    context=context,
                    evidence_type="table_row",
                    table=label,
                ))
        tables.append((label, caption, rows))

    seen = {(c.source_location, c.evidence_text) for c in candidates}
    for sec_index, (section_title, section_text) in enumerate(sections, 1):
        if not _LCI_SECTION_RE.search(section_title) and not _LCI_SECTION_RE.search(section_text[:600]):
            continue
        for line_index, line in enumerate(section_text.splitlines(), 1):
            stripped = line.lstrip("- ").strip()
            if len(stripped) < 4:
                continue
            if not (_NUMERIC_RE.search(stripped) and (_UNIT_RE.search(stripped) or line.startswith("- "))):
                continue
            location = f"section:{sec_index}:line:{line_index}"
            key = (location, stripped)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(InventoryCandidate(
                candidate_id=_stable_candidate_id(location, stripped),
                source_location=location,
                evidence_text=stripped,
                context=f"section={section_title or 'untitled'}",
                evidence_type="section_statement",
            ))

    return JATSDocument(
        doi=_article_doi(article),
        title=title,
        abstract=abstract,
        sections=sections,
        tables=tables,
        inventory_candidates=candidates,
    )

def parse_jats_file(path: Path, *, expected_doi: str | None = None) -> JATSDocument:
    return parse_jats_bytes(path.read_bytes(), expected_doi=expected_doi)
