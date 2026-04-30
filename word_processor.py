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
import uuid
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

from pii_engine import PIIEngine, AnonymizationResult

SESSIONS_DIR = Path("sessions")
SESSIONS_DIR.mkdir(exist_ok=True)


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
    """Yield all paragraphs including those inside tables."""
    # Body paragraphs
    yield from doc.paragraphs
    # Table cells
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                yield from cell.paragraphs


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

    # Collect all paragraph texts, run anonymization per paragraph,
    # and accumulate a global mapping.
    global_mapping: dict[str, str] = {}

    for para in _iter_paragraphs(doc):
        text = para.text
        if not text.strip():
            continue

        result: AnonymizationResult = engine.anonymize(text, language=language)
        if not result.mapping:
            continue

        # Build original->placeholder dict for this paragraph
        orig_to_placeholder = {v: k for k, v in result.mapping.items()}
        _replace_text_in_paragraph(para, orig_to_placeholder)
        global_mapping.update(result.mapping)  # placeholder -> original

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
