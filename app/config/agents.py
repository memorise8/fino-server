"""Agent routing configuration for FINO."""

from __future__ import annotations

from typing import Dict, List

AGENT_MAP: Dict[str, List[str]] = {
    "tax": ["TaxAgent"],
    "accounting": ["AccountingAgent"],
    "both": ["TaxAgent", "AccountingAgent"],
    "etc": ["GeneralAgent"],
}
