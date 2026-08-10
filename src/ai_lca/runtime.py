from __future__ import annotations

import os
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from typing import Literal

from .models import ExtractionProvenance


def extractor_version() -> str:
    try:
        return version("ai-lca-starter")
    except PackageNotFoundError:
        return "development"


def git_sha() -> str | None:
    for env_name in ("AI_LCA_GIT_SHA", "GITHUB_SHA"):
        value = os.getenv(env_name, "").strip()
        if value:
            return value
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = completed.stdout.strip()
    return value or None


def build_extraction_provenance(
    *,
    model: str,
    source_mode: Literal["text", "documents"],
) -> ExtractionProvenance:
    return ExtractionProvenance(
        extractor_version=extractor_version(),
        model=model,
        git_sha=git_sha(),
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        source_mode=source_mode,
    )
