"""
FastAPI application — PII-safe LLM chat interface.

Flow:
  1. User submits a message via POST /chat
  2. PIIEngine anonymizes the message (replaces PII with placeholders)
  3. Anonymized message is sent to AWS Bedrock (Claude)
  4. LLM response is restored (placeholders replaced back with original PII)
  5. Both anonymized prompt and restored response are returned to the client

Word document flow:
  POST /docx/anonymize  — upload .docx, get anonymized .docx + session_id
  POST /docx/restore    — upload anonymized .docx + session_id, get restored .docx
"""

import base64
import uuid
from pathlib import Path

from fastapi import FastAPI, Request, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from pii_engine import PIIEngine
from bedrock_client import invoke_model, MODELS, DEFAULT_MODEL_KEY
from word_processor import (
    anonymize_docx, restore_docx, list_sessions, load_session,
    review_and_comment_docx,
)
from comprehend_client import analyze_text
from dict_engine import get_dict, DICT_FILE

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

app = FastAPI(title="PII-Safe LLM Chat", version="1.0.0")
templates = Jinja2Templates(directory="templates")

# One engine instance per application (stateless — counter resets per call)
_pii_engine = PIIEngine()


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000)
    system_prompt: str = Field(
        default="You are a helpful assistant. Answer in the same language as the user.",
        max_length=2000,
    )
    language: str = Field(default="zh", pattern="^(zh|en)$")
    model_key: str = Field(default=DEFAULT_MODEL_KEY)


class PIIItem(BaseModel):
    placeholder: str
    original: str


class ChatResponse(BaseModel):
    original_message: str
    anonymized_message: str
    llm_response: str
    restored_response: str
    pii_detected: list[PIIItem]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest):
    # Step 1: Anonymize user input
    anon_result = _pii_engine.anonymize(payload.message, language=payload.language)

    # Step 2: Call LLM with anonymized text
    try:
        raw_llm_response = invoke_model(
            prompt=anon_result.anonymized_text,
            system_prompt=payload.system_prompt,
            model_key=payload.model_key,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))

    # Step 3: Restore PII in LLM response
    restored = PIIEngine.restore(raw_llm_response, anon_result.mapping)

    # Build PII summary for the UI
    pii_items = [
        PIIItem(placeholder=k, original=v)
        for k, v in anon_result.mapping.items()
    ]

    return ChatResponse(
        original_message=payload.message,
        anonymized_message=anon_result.anonymized_text,
        llm_response=raw_llm_response,
        restored_response=restored,
        pii_detected=pii_items,
    )


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/models")
async def list_models():
    return [
        {"key": k, "display_name": v.display_name}
        for k, v in MODELS.items()
    ]


# ---------------------------------------------------------------------------
# Word document routes
# ---------------------------------------------------------------------------

@app.post("/docx/anonymize")
async def docx_anonymize(
    file: UploadFile = File(...),
    language: str = Form(default="zh"),
):
    """Upload a .docx, receive anonymized .docx (base64) + session_id + mapping."""
    if not file.filename or not file.filename.lower().endswith(".docx"):
        raise HTTPException(status_code=400, detail="Only .docx files are supported.")

    uid = uuid.uuid4().hex
    input_path  = UPLOAD_DIR / f"{uid}_input.docx"
    output_path = UPLOAD_DIR / f"{uid}_anonymized.docx"

    try:
        input_path.write_bytes(await file.read())
        session_id, mapping = anonymize_docx(input_path, output_path, language=language)
        file_b64 = base64.b64encode(output_path.read_bytes()).decode()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        input_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)

    stem = Path(file.filename).stem
    return {
        "session_id":  session_id,
        "filename":    f"{stem}_anonymized.docx",
        "file_b64":    file_b64,
        "pii_detected": [
            {"placeholder": k, "original": v}
            for k, v in mapping.items()
        ],
    }


@app.post("/docx/restore")
async def docx_restore(
    file: UploadFile = File(...),
    session_id: str = Form(...),
):
    """Upload an anonymized .docx + session_id, receive restored .docx."""
    if not file.filename or not file.filename.lower().endswith(".docx"):
        raise HTTPException(status_code=400, detail="Only .docx files are supported.")

    try:
        load_session(session_id)  # validate session exists before writing files
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    uid = uuid.uuid4().hex
    input_path = UPLOAD_DIR / f"{uid}_anon.docx"
    output_path = UPLOAD_DIR / f"{uid}_restored.docx"

    try:
        input_path.write_bytes(await file.read())
        restore_docx(input_path, output_path, session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        input_path.unlink(missing_ok=True)

    stem = Path(file.filename).stem.replace("_anonymized", "")
    return FileResponse(
        path=str(output_path),
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        filename=f"{stem}_restored.docx",
    )


@app.get("/docx/sessions")
async def docx_sessions():
    """List all stored anonymization sessions."""
    return list_sessions()


@app.post("/docx/review")
async def docx_review(
    file: UploadFile = File(...),
    language: str = Form(default="zh"),
    model_key: str = Form(default=DEFAULT_MODEL_KEY),
):
    """
    One-stop pipeline: upload a .docx, anonymize it, send to the LLM for
    contract review, restore the PII, and return the restored document with
    the review suggestions attached as Word comments (批注) — plus a
    base64-encoded file for download.
    """
    if not file.filename or not file.filename.lower().endswith(".docx"):
        raise HTTPException(status_code=400, detail="Only .docx files are supported.")
    if model_key not in MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown model key '{model_key}'.")

    uid = uuid.uuid4().hex
    input_path  = UPLOAD_DIR / f"{uid}_input.docx"
    output_path = UPLOAD_DIR / f"{uid}_reviewed.docx"
    anon_path   = UPLOAD_DIR / f"{uid}_anonymized.docx"

    try:
        input_path.write_bytes(await file.read())
        result = review_and_comment_docx(
            input_path, output_path,
            model_key=model_key, language=language,
            anonymized_output_path=anon_path,
        )
        file_b64 = base64.b64encode(output_path.read_bytes()).decode()
        anon_b64 = base64.b64encode(anon_path.read_bytes()).decode()
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        input_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
        anon_path.unlink(missing_ok=True)

    stem = Path(file.filename).stem
    return {
        "session_id":   result["session_id"],
        "filename":     f"{stem}_reviewed.docx",
        "file_b64":     file_b64,
        "anonymized_filename": f"{stem}_anonymized.docx",
        "anonymized_b64":      anon_b64,
        "pii_detected": result["pii_detected"],
        "comments":     result["comments"],
        "comments_total":     result["comments_total"],
        "comments_applied":   result["comments_applied"],
        "comments_unmatched": result["comments_unmatched"],
    }


# ---------------------------------------------------------------------------
# Sensitive dictionary management routes
# ---------------------------------------------------------------------------

@app.get("/dict")
async def dict_get():
    """Return current dictionary content grouped by section."""
    d = get_dict()
    return {
        "groups": d.groups(),
        "raw": DICT_FILE.read_text(encoding="utf-8") if DICT_FILE.exists() else "",
    }


class DictSaveRequest(BaseModel):
    content: str = Field(..., max_length=500_000)


@app.post("/dict")
async def dict_save(payload: DictSaveRequest):
    """Overwrite the dictionary file and hot-reload."""
    DICT_FILE.write_text(payload.content, encoding="utf-8")
    get_dict().reload()
    return {"groups": get_dict().groups()}


# ---------------------------------------------------------------------------
# Comprehend analysis route
# ---------------------------------------------------------------------------

class ComprehendRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)
    language: str = Field(default="zh", pattern="^(zh|en)$")


@app.post("/comprehend/analyze")
async def comprehend_analyze(payload: ComprehendRequest):
    try:
        return analyze_text(payload.text, language=payload.language)
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))
