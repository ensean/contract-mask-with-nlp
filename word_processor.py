"""
Word document PII anonymization and restoration.

Key challenge: python-docx splits a paragraph into multiple Runs with
different formatting. A single PII value (e.g. "13812345678") may span
several runs. We work at the paragraph full-text level for detection,
then carefully rewrite the runs to inject placeholders while preserving
formatting as much as possible.

Session mapping is persisted as JSON under the sessions/ directory so
that a user can upload the anonymized docx later and restore it.
"""

import copy
import json
import re
import uuid
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

from pii_engine import PIIEngine, AnonymizationResult, PIISpan

SESSIONS_DIR = Path("sessions")
SESSIONS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Contract field patterns
# Extracts values after common contract field labels like "名称：XXX"
# These are used to supplement NER when spaCy misses labeled fields.
# ---------------------------------------------------------------------------

_CONTRACT_FIELD_PATTERNS: list[tuple[str, re.Pattern]] = [
    # 公司名称
    ("CN_COMPANY", re.compile(
        r"(?:甲方|乙方|丙方|委托方|受托方|采购方|销售方|供应商|买方|卖方)"
        r"(?:（[^）]*）)?"
        r"(?:名称|单位)[：:]\s*"
        r"([\u4e00-\u9fa5A-Za-z0-9（）()·&，,\s]{2,50}?)"
        r"(?=[；;，,。\n]|$|\s{2})"
    )),
    # 法定代表人 / 经营者 / 联系人
    ("PERSON", re.compile(
        r"(?:法定代表人|经营者|联系人|负责人|授权代表)[/／]?"
        r"(?:经营者|代理人)?[：:]\s*"
        r"([\u4e00-\u9fa5A-Za-z]{2,6})"
        r"(?=[；;，,。\n（\s]|$)"
    )),
    # 住所 / 地址
    ("CN_ADDRESS", re.compile(
        r"(?:住所|经营场所|地址|交货地点)[/／]?(?:经营场所)?[：:]\s*"
        r"([\u4e00-\u9fa5A-Za-z0-9（）()#\-号楼室路街道区市省]{5,80})"
        r"(?=[；;，,。\n]|$)"
    )),
]


def _extract_contract_field_spans(text: str) -> list[PIISpan]:
    """Extract PII spans from labeled contract fields (e.g. '名称：XX公司')."""
    spans: list[PIISpan] = []
    for pii_type, pattern in _CONTRACT_FIELD_PATTERNS:
        for m in pattern.finditer(text):
            value = m.group(1).strip()
            if not value:
                continue
            start = m.start(1)
            end = start + len(value)
            spans.append(PIISpan(
                start=start, end=end,
                pii_type=pii_type,
                original=value,
            ))
    return spans


# ---------------------------------------------------------------------------
# Session persistence
# ---------------------------------------------------------------------------

def save_session(mapping: dict[str, str]) -> str:
    """Persist *mapping* to disk and return the session_id."""
    session_id = uuid.uuid4().hex
    path = SESSIONS_DIR / f"{session_id}.json"
    path.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    return session_id


def load_session(session_id: str) -> dict[str, str]:
    """Load and return the mapping for *session_id*. Raises FileNotFoundError if missing."""
    path = SESSIONS_DIR / f"{session_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Session '{session_id}' not found.")
    return json.loads(path.read_text(encoding="utf-8"))


def list_sessions() -> list[dict]:
    """Return metadata for all stored sessions."""
    result = []
    for p in sorted(SESSIONS_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            mapping = json.loads(p.read_text(encoding="utf-8"))
            result.append({
                "session_id": p.stem,
                "pii_count": len(mapping),
                "modified": p.stat().st_mtime,
            })
        except Exception:
            pass
    return result


# ---------------------------------------------------------------------------
# Run-level text replacement inside a paragraph
# ---------------------------------------------------------------------------

def _replace_text_in_paragraph(paragraph, replacements: dict[str, str]) -> None:
    """
    Apply *replacements* (original -> placeholder) to *paragraph* while
    preserving run-level formatting.

    Strategy:
      1. Collect the full paragraph text and compute replacement positions.
      2. Rebuild the run list: the first run absorbs all text (with its
         original formatting), subsequent runs are cleared.
      3. For spans that cross run boundaries we merge into the first run.
    """
    if not paragraph.runs:
        return

    # Build full text and a map: char_index -> run_index
    full_text = paragraph.text
    if not full_text.strip():
        return

    # Apply all replacements to the full text
    new_text = full_text
    for original, placeholder in replacements.items():
        new_text = new_text.replace(original, placeholder)

    if new_text == full_text:
        return  # nothing changed

    # Rewrite: put all new text into the first run, clear the rest
    first_run = paragraph.runs[0]
    first_run.text = new_text
    for run in paragraph.runs[1:]:
        run.text = ""


def _replace_text_in_paragraph_restore(paragraph, mapping: dict[str, str]) -> None:
    """Restore placeholders -> originals in a paragraph."""
    _replace_text_in_paragraph(paragraph, mapping)


# ---------------------------------------------------------------------------
# Document-level anonymization
# ---------------------------------------------------------------------------

def _iter_paragraphs(doc: Document):
    """
    Yield all paragraphs in the document, including those inside:
    - Regular body
    - Tables (all cells)
    - Structured Document Tags / Content Controls (w:sdtContent)
      — these are commonly used in contract templates for fillable fields
    """
    from docx.oxml.ns import qn
    from docx.text.paragraph import Paragraph as DocxParagraph

    def _paragraphs_from_element(element):
        """Recursively yield Paragraph objects from an XML element."""
        for child in element:
            if child.tag == qn("w:p"):
                yield DocxParagraph(child, element)
            elif child.tag in (qn("w:sdtContent"), qn("w:tbl"),
                               qn("w:tr"), qn("w:tc"), qn("w:sdt")):
                yield from _paragraphs_from_element(child)

    # Body paragraphs (includes sdtContent via recursive walk)
    yield from _paragraphs_from_element(doc.element.body)

    # Table cells (belt-and-suspenders for nested tables)
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for para in cell.paragraphs:
                    yield para


def anonymize_docx(
    input_path: str | Path,
    output_path: str | Path,
    language: str = "zh",
) -> tuple[str, dict[str, str]]:
    """
    Read *input_path*, anonymize all PII, write anonymized docx to
    *output_path*.

    Returns (session_id, mapping).
    """
    doc = Document(str(input_path))
    engine = PIIEngine()
    engine._reset_counter()   # single document-level counter, never reset again

    global_mapping: dict[str, str] = {}

    for para in _iter_paragraphs(doc):
        text = para.text
        if not text.strip():
            continue

        # Step 1: Contract field extraction (highest priority, label-anchored)
        field_spans = _extract_contract_field_spans(text)

        # Step 2: General PII engine with document-level counter (no reset)
        result: AnonymizationResult = engine._anonymize_no_reset(text, language=language)

        # Step 3: Merge — field spans add values not already found by engine
        combined: dict[str, str] = {}  # original -> placeholder
        for placeholder, original in result.mapping.items():
            combined[original] = placeholder
        for span in field_spans:
            val = span.original
            if val and val not in combined:
                placeholder = engine._next_placeholder(span.pii_type)
                combined[val] = placeholder

        if not combined:
            continue

        para_mapping = {ph: orig for orig, ph in combined.items()}
        _replace_text_in_paragraph(para, combined)
        global_mapping.update(para_mapping)

    doc.save(str(output_path))
    session_id = save_session(global_mapping)
    return session_id, global_mapping


# ---------------------------------------------------------------------------
# Document-level restoration
# ---------------------------------------------------------------------------

def restore_docx(
    input_path: str | Path,
    output_path: str | Path,
    session_id: str,
) -> None:
    """
    Read anonymized *input_path*, restore PII using *session_id* mapping,
    write restored docx to *output_path*.
    """
    mapping = load_session(session_id)  # placeholder -> original
    doc = Document(str(input_path))

    for para in _iter_paragraphs(doc):
        if not para.text.strip():
            continue
        _replace_text_in_paragraph_restore(para, mapping)

    doc.save(str(output_path))
