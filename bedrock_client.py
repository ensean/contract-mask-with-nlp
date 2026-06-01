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

import logging

import boto3
from botocore.exceptions import ClientError
from typing import NamedTuple


logger = logging.getLogger("pii.bedrock")

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
    supports_temperature: bool = True


MODELS: dict[str, ModelInfo] = {
    "claude-sonnet-4-6": ModelInfo(
        model_id="global.anthropic.claude-sonnet-4-6",
        display_name="Claude Sonnet 4.6 (Anthropic)",
        max_tokens=8192,
    ),
    "claude-opus-4-8": ModelInfo(
        model_id="global.anthropic.claude-opus-4-8",
        display_name="Claude Opus 4.8 (Anthropic)",
        max_tokens=8192,
        # Opus 4.7+ deprecated sampling params (temperature/top_p/top_k);
        # passing temperature triggers a ValidationException.
        supports_temperature=False,
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


def _build_inference_config(info: ModelInfo, temperature: float) -> dict:
    """
    Build the Converse inferenceConfig, omitting *temperature* for models that
    have deprecated sampling parameters (e.g. Claude Opus 4.7+).
    """
    config: dict = {"maxTokens": info.max_tokens}
    if info.supports_temperature:
        config["temperature"] = temperature
    return config


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
            inferenceConfig=_build_inference_config(info, temperature),
        )
        # Some models (e.g. MiniMax M2.5) prepend a reasoningContent block
        # before the actual text block — find the first text block.
        contents = response["output"]["message"]["content"]
        for block in contents:
            if "text" in block:
                return block["text"]
        raise RuntimeError("No text block found in Bedrock response content")
    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        error_msg = e.response["Error"]["Message"]
        raise RuntimeError(f"Bedrock API error [{error_code}]: {error_msg}") from e
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"Unexpected Bedrock response format: {e}") from e


# ---------------------------------------------------------------------------
# Contract review (returns structured comment suggestions)
# ---------------------------------------------------------------------------

_REVIEW_SYSTEM_PROMPT = (
    "你是一名资深合同审阅律师。你会收到一份合同文本，其中的敏感信息（公司名、人名、"
    "电话、账号、地址等）已在发给你之前被系统替换为 <<TYPE_N>> 形式的占位符"
    "（如 <<CN_COMPANY_1>>、<<PERSON_1>>、<<CN_PHONE_1>>）。\n\n"
    "【关于占位符 —— 非常重要，必须遵守】\n"
    "1. 占位符只是脱敏处理的技术产物，代表合同中真实存在、且完全正常的敏感信息。"
    "请把它当作正常的具体值来阅读和审阅，就当它是一个普通的公司名/人名/号码。\n"
    "2. 【严禁】对脱敏本身或占位符的格式提出任何意见，例如不要说"
    "“此处为占位符”“信息被掩码”“建议填写完整信息”“XX 信息缺失/不清”之类的话——"
    "这些都是脱敏造成的假象，不是合同的真实问题。\n"
    "3. 【严禁】因为两个占位符编号不同（如 <<CN_COMPANY_2>> 与 <<CN_COMPANY_4>>）"
    "就推断它们代表不同主体，也不要因为编号相同就断定是同一主体。占位符的编号与原文是否"
    "同一实体【没有必然关系】，请勿据此判断主体一致性、账户归属等问题。\n"
    "4. 你只审阅合同条款本身的法律与商业风险（如违约责任、付款条件、交付、知识产权、"
    "保密、争议解决等），把占位符背后的实体视为已知且正常，聚焦于条款逻辑而非被脱敏的具体值。\n\n"
    "请逐条审阅合同，找出存在风险、表述不清、缺失条款或对一方不利的地方，"
    "并以 Word 批注的形式给出修改建议。\n\n"
    "【输出格式 —— 必须严格遵守】\n"
    "为每一条批注输出一个块，格式如下（标记必须独占一行，前后不要加任何其它字符）：\n"
    "@@QUOTE@@\n"
    "<合同原文中需要被批注的片段>\n"
    "@@COMMENT@@\n"
    "<针对该片段的审阅意见和修改建议>\n"
    "@@END@@\n\n"
    "多条批注就重复多个这样的块。除这些块之外，不要输出任何前言、总结、"
    "Markdown 代码块标记或解释性文字。如果合同没有任何问题，只输出一行：@@NONE@@\n\n"
    "【内容要求】\n"
    "1. QUOTE 必须从合同原文中【一字不差地逐字复制】，包括其中的占位符；"
    "必须是单行、连续、不跨段落的文本，建议 5-30 字，且在全文中尽量唯一，否则无法定位批注。\n"
    "2. QUOTE 优先选取关键短语而非整段或整句。\n"
    "3. 占位符 <<TYPE_N>> 视为不可改动的整体，可出现在 QUOTE 中，也可在 COMMENT 中引用；"
    "但 COMMENT 的意见不得针对占位符或脱敏行为本身。\n"
    "4. COMMENT 可以分多行书写，内容要具体、专业、可操作；COMMENT 中可以自由使用引号、"
    "标点等任何字符，无需转义。\n"
    "5. 不要在 QUOTE 或 COMMENT 的正文里出现 @@QUOTE@@、@@COMMENT@@、@@END@@ 这些标记字样。"
)


class ReviewComment(NamedTuple):
    quote: str
    comment: str


def _parse_review_blocks(text: str) -> list["ReviewComment"]:
    """
    Parse the delimiter-based review format into ReviewComment list.

    Expected format (repeated):
        @@QUOTE@@
        <quote, single line>
        @@COMMENT@@
        <comment, may span multiple lines>
        @@END@@

    Robust to quotes/newlines/punctuation inside the comment body, which is
    why we avoid JSON (LLMs routinely emit unescaped " inside Chinese text).
    Tolerates a missing trailing @@END@@ on the last block.
    """
    if "@@NONE@@" in text and "@@QUOTE@@" not in text:
        return []

    comments: list[ReviewComment] = []
    # Each record starts at @@QUOTE@@
    for chunk in text.split("@@QUOTE@@")[1:]:
        if "@@COMMENT@@" not in chunk:
            continue
        quote_part, rest = chunk.split("@@COMMENT@@", 1)
        # Comment ends at @@END@@ if present, else at the next record / string end
        comment_part = rest.split("@@END@@", 1)[0]
        quote = quote_part.strip()
        comment = comment_part.strip()
        if quote and comment:
            comments.append(ReviewComment(quote=quote, comment=comment))
    return comments


def review_contract(
    contract_text: str,
    model_key: str = DEFAULT_MODEL_KEY,
    temperature: float = 0.3,
) -> list[ReviewComment]:
    """
    Send (already anonymized) *contract_text* to Bedrock and parse the model's
    structured review into a list of ReviewComment(quote, comment).

    Uses a delimiter-based output format (not JSON) so that quotes, newlines
    and punctuation inside the model's Chinese commentary cannot corrupt
    parsing.

    Raises RuntimeError on API failure or unknown model key.
    Returns an empty list if the model finds nothing or returns unparseable output.
    """
    if model_key not in MODELS:
        raise RuntimeError(
            f"Unknown model key '{model_key}'. "
            f"Valid options: {list(MODELS.keys())}"
        )

    info = MODELS[model_key]
    client = _get_client()

    try:
        response = client.converse(
            modelId=info.model_id,
            system=[{"text": _REVIEW_SYSTEM_PROMPT}],
            messages=[{"role": "user", "content": [{"text": contract_text}]}],
            inferenceConfig=_build_inference_config(info, temperature),
        )
        contents = response["output"]["message"]["content"]
        raw = next((b["text"] for b in contents if "text" in b), None)
        if raw is None:
            raise RuntimeError("No text block found in Bedrock response content")
    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        error_msg = e.response["Error"]["Message"]
        raise RuntimeError(f"Bedrock API error [{error_code}]: {error_msg}") from e
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"Unexpected Bedrock response format: {e}") from e

    comments = _parse_review_blocks(raw)
    if not comments and "@@NONE@@" not in raw:
        logger.warning(
            "review_contract: model output produced no parseable comments. "
            "Raw response (first 1000 chars):\n%s",
            raw[:1000],
        )
    else:
        logger.info("review_contract: parsed %d comment(s) from model output.", len(comments))
    return comments
