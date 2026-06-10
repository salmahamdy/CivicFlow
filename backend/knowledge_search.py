"""
Keyword-based search over the business regulations knowledge base.
Returns matching regulations ranked by relevance score.
"""

import json
import os
from typing import List, Optional

_KB_PATH = os.path.join(os.path.dirname(__file__), "knowledge_base.json")
_kb_cache: Optional[dict] = None


def _load_kb() -> dict:
    global _kb_cache
    if _kb_cache is None:
        with open(_KB_PATH, "r") as f:
            _kb_cache = json.load(f)
    return _kb_cache


def _tokenize(text: str) -> set:
    """Lowercase split + strip punctuation."""
    return {
        w.strip(".,;:!?()[]{}\"'")
        for w in text.lower().split()
        if len(w.strip(".,;:!?()[]{}\"'")) > 1
    }


def search_regulations(query: str, top_k: int = 2) -> List[dict]:
    """
    Search the knowledge base by keyword matching.

    Scoring:
      - Exact phrase match on a keyword     → +10
      - Individual word overlap with keywords → +2 each
      - Word found in title or additional_rules → +1 each

    Returns top_k regulations sorted by score, with general
    requirements appended.
    """
    kb = _load_kb()
    query_lower = query.lower()
    query_tokens = _tokenize(query)

    scored = []

    for reg in kb["regulations"]:
        score = 0

        # Phrase match against keyword list (highest signal)
        for kw in reg["keywords"]:
            if kw in query_lower:
                score += 10

        # Token overlap with keywords
        kw_tokens = set()
        for kw in reg["keywords"]:
            kw_tokens.update(_tokenize(kw))
        overlap = query_tokens & kw_tokens
        score += len(overlap) * 2

        # Token match in title and rules (lower signal)
        title_tokens = _tokenize(reg["title"])
        rules_tokens = _tokenize(reg.get("additional_rules", ""))
        score += len(query_tokens & title_tokens)
        score += len(query_tokens & rules_tokens)

        if score > 0:
            scored.append((score, reg))

    # Sort by score descending, take top_k
    scored.sort(key=lambda x: x[0], reverse=True)
    results = [reg for _, reg in scored[:top_k]]

    # Attach general requirements to each result
    general = kb.get("general_requirements", {})
    for r in results:
        r["general_requirements"] = general

    return results


def format_results(results: List[dict]) -> str:
    """Format search results into readable text for the LLM."""
    if not results:
        return "No matching regulations found. A general business license (fee: $50) is likely required. Please verify with local authorities."

    parts = []
    for reg in results:
        permits_str = "\n".join(
            f"  - {p['name']}: ${p['fee']} (processing: {p['processing_days']} days)"
            for p in reg["permits"]
        )
        docs_str = "\n".join(f"  - {d}" for d in reg["required_documents"])

        parts.append(
            f"[{reg['id']}] {reg['title']}\n"
            f"Business type: {reg['business_type']}\n\n"
            f"Required permits:\n{permits_str}\n\n"
            f"Required documents:\n{docs_str}\n\n"
            f"Zoning: {reg['zoning_restrictions']}\n\n"
            f"Additional rules: {reg['additional_rules']}"
        )

    # Append general requirements once
    general = results[0].get("general_requirements", {})
    if general:
        general_rules = "\n".join(f"  - {r}" for r in general.get("all_businesses", []))
        parts.append(
            f"\nGeneral requirements (all businesses):\n{general_rules}\n"
            f"Note: {general.get('fee_notes', '')}\n"
            f"Note: {general.get('processing_note', '')}"
        )

    return "\n\n---\n\n".join(parts)
