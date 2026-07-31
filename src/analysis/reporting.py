
from __future__ import annotations

from typing import Any, Sequence


PHASH_METHOD = "OpenCV pHash"
PROPOSED_METHOD = "Proposed hybrid authentication workflow"
PRIMARY_METHODS = (PHASH_METHOD, PROPOSED_METHOD)


def validate_primary_method_comparison(rows: Sequence[dict[str, Any]]) -> None:

    methods = tuple(str(row.get("method")) for row in rows)
    if methods != PRIMARY_METHODS:
        raise ValueError(
            f"Primary comparison must contain {PRIMARY_METHODS}, in order; got {methods}."
        )
    if any("sha-256" in method.lower() or "sha256" in method.lower() for method in methods):
        raise ValueError("SHA-256 must not be selected as the primary paper baseline.")
