"""
Bilingual (English + Arabic) keyword search over the Egyptian business
regulations knowledge base. Returns matching regulations ranked by score.

Arabic support:
- Normalizes common letter variants (أ/إ/آ -> ا, ة -> ه, ى -> ي)
- Strips tashkeel (diacritics)
- Matches against the keywords_ar list in each KB entry
"""

import json
import os
import re
from typing import List, Optional

_KB_PATH = os.path.join(os.path.dirname(__file__), "knowledge_base.json")
_kb_cache: Optional[dict] = None

_TASHKEEL = re.compile(r"[\u0617-\u061A\u064B-\u0652]")


def _load_kb() -> dict:
    global _kb_cache
    if _kb_cache is None:
        with open(_KB_PATH, "r", encoding="utf-8") as f:
            _kb_cache = json.load(f)
    return _kb_cache


def normalize_arabic(text: str) -> str:
    """Normalize Arabic letter variants so e.g. حضانه matches حضانة."""
    text = _TASHKEEL.sub("", text)
    text = re.sub("[إأآا]", "ا", text)
    text = text.replace("ة", "ه")
    text = text.replace("ى", "ي")
    text = text.replace("ؤ", "و")
    text = text.replace("ئ", "ي")
    return text


def _normalize(text: str) -> str:
    """Lowercase (affects Latin only) + Arabic normalization."""
    return normalize_arabic(text.lower())


def _tokenize(text: str) -> set:
    """Split on non-word characters, normalize, drop 1-char tokens."""
    tokens = re.split(r"[\s.,;:!?()\[\]{}\"'،؛؟]+", _normalize(text))
    return {t for t in tokens if len(t) > 1}


def search_regulations(query: str, top_k: int = 2) -> List[dict]:
    """
    Search the knowledge base by bilingual keyword matching.

    Scoring:
      - Exact phrase match on an EN or AR keyword      -> +10
      - Individual word overlap with EN or AR keywords -> +2 each
      - Word found in title or additional_rules        -> +1 each
    """
    kb = _load_kb()
    query_norm = _normalize(query)
    query_tokens = _tokenize(query)

    scored = []

    for reg in kb["regulations"]:
        score = 0
        all_keywords = reg.get("keywords", []) + reg.get("keywords_ar", [])

        # Phrase match against both keyword lists (highest signal)
        for kw in all_keywords:
            if _normalize(kw) in query_norm:
                score += 10

        # Token overlap with keywords
        kw_tokens = set()
        for kw in all_keywords:
            kw_tokens.update(_tokenize(kw))
        score += len(query_tokens & kw_tokens) * 2

        # Token match in title and rules (lower signal)
        score += len(query_tokens & _tokenize(reg["title"]))
        score += len(query_tokens & _tokenize(reg.get("additional_rules", "")))

        if score > 0:
            scored.append((score, reg))

    scored.sort(key=lambda x: x[0], reverse=True)
    results = [reg for _, reg in scored[:top_k]]

    general = kb.get("general_requirements", {})
    for r in results:
        r["general_requirements"] = general

    return results


def format_results(results: List[dict]) -> str:
    """Format search results into readable text for the LLM."""
    if not results:
        return (
            "No specific regulations matched. At minimum, an Egyptian business "
            "needs: Commercial Registry entry (السجل التجاري), a Tax Card "
            "(البطاقة الضريبية) from the Egyptian Tax Authority, and VAT "
            "registration if annual turnover exceeds EGP 500,000. "
            "Advise the applicant to verify requirements with GAFI and the "
            "local district office."
        )

    parts = []
    for reg in results:
        permits_str = "\n".join(
            f"  - {p['name']}: EGP {p['fee_egp']} (processing: ~{p['processing_days']} days)"
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

    general = results[0].get("general_requirements", {})
    if general:
        general_rules = "\n".join(f"  - {r}" for r in general.get("all_businesses", []))
        parts.append(
            f"\nGeneral requirements (all Egyptian businesses):\n{general_rules}\n"
            f"Note: {general.get('fee_notes', '')}\n"
            f"Note: {general.get('processing_note', '')}"
        )

    kb = _load_kb()
    parts.append(f"\nDISCLAIMER: {kb.get('disclaimer', '')}")

    return "\n\n---\n\n".join(parts)
