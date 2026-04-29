"""
FastAPI application — PII-safe LLM chat interface.

Flow:
  1. User submits a message via POST /chat
  2. PIIEngine anonymizes the message (replaces PII with placeholders)
  3. Anonymized message is sent to AWS Bedrock (Claude)
  4. LLM response is restored (placeholders replaced back with original PII)
  5. Both anonymized prompt and restored response are returned to the client
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from pii_engine import PIIEngine
from bedrock_client import invoke_model, MODELS, DEFAULT_MODEL_KEY

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
