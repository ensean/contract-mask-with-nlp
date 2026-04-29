"""
PII Detection and Anonymization Engine
Supports: Chinese phone numbers, ID cards, bank cards, emails, names (via spaCy NER),
          addresses, and English equivalents via Presidio.
"""

import re
import uuid
from dataclasses import dataclass, field
from typing import Optional

from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_analyzer.nlp_engine import NlpEngineProvider


# ---------------------------------------------------------------------------
# Custom regex patterns for Chinese PII
# ---------------------------------------------------------------------------

PATTERNS = {
    "CN_PHONE": re.compile(
        r"(?<!\d)"
        r"((\+?86[-\s]?)?"
        r"(1[3-9]\d{9}))"
        r"(?!\d)"
    ),
    "CN_ID_CARD": re.compile(
        r"(?<!\d)"
        r"[1-9]\d{5}"          # region code
        r"(19|20)\d{2}"        # year
        r"(0[1-9]|1[0-2])"    # month
        r"(0[1-9]|[12]\d|3[01])"  # day
        r"\d{3}[\dXx]"
        r"(?!\d)"
    ),
    "CN_BANK_CARD": re.compile(
        r"(?<!\d)"
        r"[3-9]\d{15,18}"
        r"(?!\d)"
    ),
    "EMAIL": re.compile(
        r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"
    ),
    "CN_ADDRESS": re.compile(
        r"[\u4e00-\u9fa5]{2,6}(?:省|自治区|市)"
        r"[\u4e00-\u9fa5]{2,10}(?:市|区|县|镇|乡)"
        r"[\u4e00-\u9fa5\d]{2,30}(?:路|街|道|巷|弄|号|楼|室|单元)[\d\-A-Za-z]*"
    ),
}


# ---------------------------------------------------------------------------
# Presidio setup (handles English PII + supplements Chinese)
# ---------------------------------------------------------------------------

def _build_analyzer() -> AnalyzerEngine:
    """Build a Presidio AnalyzerEngine with both zh and en NLP models."""
    configuration = {
        "nlp_engine_name": "spacy",
        "models": [
            {"lang_code": "zh", "model_name": "zh_core_web_sm"},
            {"lang_code": "en", "model_name": "en_core_web_sm"},
        ],
    }
    provider = NlpEngineProvider(nlp_configuration=configuration)
    nlp_engine = provider.create_engine()
    registry = RecognizerRegistry()
    registry.load_predefined_recognizers(nlp_engine=nlp_engine)
    return AnalyzerEngine(nlp_engine=nlp_engine, registry=registry)


_analyzer: Optional[AnalyzerEngine] = None


def get_analyzer() -> AnalyzerEngine:
    global _analyzer
    if _analyzer is None:
        _analyzer = _build_analyzer()
    return _analyzer


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------

@dataclass
class PIISpan:
    start: int
    end: int
    pii_type: str
    original: str
    placeholder: str = ""


@dataclass
class AnonymizationResult:
    anonymized_text: str
    mapping: dict[str, str] = field(default_factory=dict)   # placeholder -> original
    spans: list[PIISpan] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class PIIEngine:
    """
    Detect PII in text, replace with placeholders, and restore later.

    Usage:
        engine = PIIEngine()
        result = engine.anonymize("我叫张三，手机 13812345678")
        restored = engine.restore(llm_response, result.mapping)
    """

    def __init__(self):
        self._counter: dict[str, int] = {}

    def _next_placeholder(self, pii_type: str) -> str:
        self._counter[pii_type] = self._counter.get(pii_type, 0) + 1
        return f"<<{pii_type}_{self._counter[pii_type]}>>"

    def _reset_counter(self):
        self._counter = {}

    # ------------------------------------------------------------------
    # Regex-based detection
    # ------------------------------------------------------------------

    def _regex_spans(self, text: str) -> list[PIISpan]:
        spans: list[PIISpan] = []
        for pii_type, pattern in PATTERNS.items():
            for m in pattern.finditer(text):
                spans.append(PIISpan(
                    start=m.start(),
                    end=m.end(),
                    pii_type=pii_type,
                    original=m.group(),
                ))
        return spans

    # ------------------------------------------------------------------
    # Presidio-based detection
    # ------------------------------------------------------------------

    def _presidio_spans(self, text: str, language: str = "zh") -> list[PIISpan]:
        analyzer = get_analyzer()
        try:
            results = analyzer.analyze(
                text=text,
                language=language,
                entities=[
                    "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER",
                    "CREDIT_CARD", "IBAN_CODE", "IP_ADDRESS",
                    "LOCATION", "DATE_TIME",
                ],
            )
        except Exception:
            results = []

        spans: list[PIISpan] = []
        for r in results:
            pii_type = r.entity_type
            spans.append(PIISpan(
                start=r.start,
                end=r.end,
                pii_type=pii_type,
                original=text[r.start:r.end],
            ))
        return spans

    # ------------------------------------------------------------------
    # Merge & deduplicate overlapping spans
    # Regex spans take priority over Presidio NER spans when overlapping.
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_spans(
        regex_spans: list[PIISpan],
        presidio_spans: list[PIISpan],
    ) -> list[PIISpan]:
        """
        Merge regex and presidio spans.
        Regex spans have higher priority: any presidio span that overlaps
        with a regex span is discarded.
        """
        # Build a set of (start, end) ranges covered by regex
        regex_ranges = [(s.start, s.end) for s in regex_spans]

        def overlaps_regex(span: PIISpan) -> bool:
            for rs, re_ in regex_ranges:
                if span.start < re_ and span.end > rs:
                    return True
            return False

        filtered_presidio = [s for s in presidio_spans if not overlaps_regex(s)]
        all_spans = regex_spans + filtered_presidio

        if not all_spans:
            return []

        # Sort by start; prefer longer spans on tie
        sorted_spans = sorted(all_spans, key=lambda s: (s.start, -(s.end - s.start)))
        merged: list[PIISpan] = [sorted_spans[0]]
        for span in sorted_spans[1:]:
            last = merged[-1]
            if span.start < last.end:   # overlapping — skip
                continue
            merged.append(span)
        return merged

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def anonymize(self, text: str, language: str = "zh") -> AnonymizationResult:
        """Replace PII in *text* with placeholders. Returns anonymized text + mapping."""
        self._reset_counter()

        regex_spans = self._regex_spans(text)
        presidio_spans = self._presidio_spans(text, language)
        all_spans = self._merge_spans(regex_spans, presidio_spans)

        mapping: dict[str, str] = {}
        result_chars = list(text)
        offset = 0  # track character shift after replacements

        # Work on original positions, apply replacements left-to-right
        for span in all_spans:
            placeholder = self._next_placeholder(span.pii_type)
            span.placeholder = placeholder
            mapping[placeholder] = span.original

        # Rebuild text with placeholders (right-to-left to preserve indices)
        anonymized = text
        for span in reversed(all_spans):
            anonymized = anonymized[: span.start] + span.placeholder + anonymized[span.end :]

        return AnonymizationResult(
            anonymized_text=anonymized,
            mapping=mapping,
            spans=all_spans,
        )

    @staticmethod
    def restore(text: str, mapping: dict[str, str]) -> str:
        """Replace placeholders in *text* back with original PII values."""
        for placeholder, original in mapping.items():
            text = text.replace(placeholder, original)
        return text
