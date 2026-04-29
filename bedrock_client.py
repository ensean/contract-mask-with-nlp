"""
AWS Bedrock client wrapper.
Uses the 'test' AWS CLI profile.

Supported models:
  - Claude Sonnet 4.6  (Anthropic)   — anthropic.claude-sonnet-4-6
  - Kimi K2.5          (Moonshot AI) — moonshotai.kimi-k2.5
  - GLM 5              (Z.AI)        — zai.glm-5
  - MiniMax M2.5       (MiniMax)     — minimax.minimax-m2.5

Kimi / GLM / MiniMax use the OpenAI-compatible Converse API body format,
while Claude uses its own anthropic_version body format.
We normalise everything through boto3's converse() call which works for all.
"""

import boto3
from botocore.exceptions import ClientError
from typing import NamedTuple


AWS_REGION = "ap-northeast-1"
AWS_PROFILE = "test"

PII_PLACEHOLDER_INSTRUCTION = (
    "The user's message may contain placeholders in the format <<TYPE_N>> "
    "(e.g. <<PERSON_1>>, <<CN_PHONE_2>>). "
    "These placeholders represent sensitive information that has been masked. "
    "You MUST treat them as opaque tokens: preserve them exactly as-is in your response "
    "whenever you refer to the corresponding entity. "
    "Do NOT expand, translate, paraphrase, remove, or alter these placeholders in any way."
)


# ---------------------------------------------------------------------------
# Model catalogue
# ---------------------------------------------------------------------------

class ModelInfo(NamedTuple):
    model_id: str
    display_name: str
    max_tokens: int


MODELS: dict[str, ModelInfo] = {
    "claude-sonnet-4-6": ModelInfo(
        model_id="global.anthropic.claude-sonnet-4-6",
        display_name="Claude Sonnet 4.6 (Anthropic)",
        max_tokens=8192,
    ),
    "kimi-k2.5": ModelInfo(
        model_id="moonshotai.kimi-k2.5",
        display_name="Kimi K2.5 (Moonshot AI)",
        max_tokens=8192,
    ),
    "glm-5": ModelInfo(
        model_id="zai.glm-5",
        display_name="GLM 5 (Z.AI)",
        max_tokens=8192,
    ),
    "minimax-m2.5": ModelInfo(
        model_id="minimax.minimax-m2.5",
        display_name="MiniMax M2.5 (MiniMax)",
        max_tokens=8192,
    ),
}

DEFAULT_MODEL_KEY = "claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

def _get_client():
    session = boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)
    return session.client("bedrock-runtime")


def invoke_model(
    prompt: str,
    system_prompt: str = "You are a helpful assistant. Answer in the same language as the user.",
    model_key: str = DEFAULT_MODEL_KEY,
    temperature: float = 0.7,
) -> str:
    """
    Send *prompt* to Bedrock using the Converse API (works for all providers).
    *model_key* must be one of the keys in MODELS.
    Raises RuntimeError on API failure or unknown model key.
    """
    if model_key not in MODELS:
        raise RuntimeError(
            f"Unknown model key '{model_key}'. "
            f"Valid options: {list(MODELS.keys())}"
        )

    info = MODELS[model_key]
    client = _get_client()

    full_system = f"{system_prompt}\n\n{PII_PLACEHOLDER_INSTRUCTION}"

    try:
        response = client.converse(
            modelId=info.model_id,
            system=[{"text": full_system}],
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={
                "maxTokens": info.max_tokens,
                "temperature": temperature,
            },
        )
        return response["output"]["message"]["content"][0]["text"]
    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        error_msg = e.response["Error"]["Message"]
        raise RuntimeError(f"Bedrock API error [{error_code}]: {error_msg}") from e
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"Unexpected Bedrock response format: {e}") from e
