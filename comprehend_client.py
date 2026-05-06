"""
AWS Comprehend client wrapper.
Uses the 'test' AWS CLI profile.

Capabilities used:
  - detect_entities:     Supports all languages incl. zh — returns PERSON, ORG,
                         LOCATION, DATE, QUANTITY, TITLE, EVENT, etc.
  - detect_pii_entities: English / Spanish only — returns NAME, PHONE, EMAIL,
                         CREDIT_DEBIT_NUMBER, ADDRESS, SSN, etc.
  - detect_key_phrases:  All supported languages — returns salient phrases.

For Chinese contract text we call detect_entities (language_code="zh").
For English text we additionally call detect_pii_entities.
"""

import boto3
from botocore.exceptions import ClientError

AWS_REGION = "us-east-1"
AWS_PROFILE = "test"

# Comprehend entity type → human-readable label (Chinese)
ENTITY_TYPE_LABELS: dict[str, str] = {
    "PERSON":           "人名",
    "LOCATION":         "地点",
    "ORGANIZATION":     "组织/公司",
    "COMMERCIAL_ITEM":  "商品",
    "EVENT":            "事件",
    "DATE":             "日期",
    "QUANTITY":         "数量",
    "TITLE":            "标题/职位",
    "OTHER":            "其他",
}

PII_TYPE_LABELS: dict[str, str] = {
    "NAME":                  "姓名",
    "PHONE":                 "电话",
    "EMAIL":                 "邮箱",
    "ADDRESS":               "地址",
    "CREDIT_DEBIT_NUMBER":   "信用/借记卡号",
    "CREDIT_DEBIT_CVV":      "CVV",
    "CREDIT_DEBIT_EXPIRY":   "卡有效期",
    "BANK_ACCOUNT_NUMBER":   "银行账号",
    "BANK_ROUTING":          "银行路由号",
    "SSN":                   "社会安全号",
    "PASSPORT_NUMBER":       "护照号",
    "DRIVER_ID":             "驾照号",
    "DATE_TIME":             "日期时间",
    "AGE":                   "年龄",
    "URL":                   "网址",
    "IP_ADDRESS":            "IP地址",
    "MAC_ADDRESS":           "MAC地址",
    "USERNAME":              "用户名",
    "PASSWORD":              "密码",
    "AWS_ACCESS_KEY":        "AWS访问密钥",
    "AWS_SECRET_KEY":        "AWS密钥",
    "PIN":                   "PIN码",
    "SWIFT_CODE":            "SWIFT代码",
    "VEHICLE_IDENTIFICATION_NUMBER": "车辆识别码",
    "LICENSE_PLATE":         "车牌号",
    "UK_NATIONAL_INSURANCE_NUMBER":  "英国国民保险号",
    "IN_AADHAAR":            "印度Aadhaar号",
}


def _get_client():
    session = boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)
    return session.client("comprehend")


def analyze_text(text: str, language: str = "zh") -> dict:
    """
    Run Comprehend analysis on *text* and return a structured result dict:
    {
        "entities":     [ {text, type, type_label, score, begin, end}, ... ],
        "pii_entities": [ {text, type, type_label, score, begin, end}, ... ],  # en only
        "key_phrases":  [ {text, score, begin, end}, ... ],
        "language":     "zh" | "en",
        "pii_supported": bool,
    }
    Raises RuntimeError on API failure.
    """
    client = _get_client()

    # Comprehend language codes
    lang_code = "zh" if language == "zh" else "en"
    pii_supported = lang_code == "en"

    result: dict = {
        "entities": [],
        "pii_entities": [],
        "key_phrases": [],
        "language": lang_code,
        "pii_supported": pii_supported,
    }

    # Truncate to Comprehend's 5000 UTF-8 byte limit
    encoded = text.encode("utf-8")
    if len(encoded) > 4900:
        text = encoded[:4900].decode("utf-8", errors="ignore")

    try:
        # 1. Entity detection (all languages)
        ent_resp = client.detect_entities(Text=text, LanguageCode=lang_code)
        for e in ent_resp.get("Entities", []):
            result["entities"].append({
                "text":       text[e["BeginOffset"]: e["EndOffset"]],
                "type":       e["Type"],
                "type_label": ENTITY_TYPE_LABELS.get(e["Type"], e["Type"]),
                "score":      round(e["Score"], 4),
                "begin":      e["BeginOffset"],
                "end":        e["EndOffset"],
            })

        # 2. PII detection (English / Spanish only)
        if pii_supported:
            pii_resp = client.detect_pii_entities(Text=text, LanguageCode=lang_code)
            for p in pii_resp.get("Entities", []):
                result["pii_entities"].append({
                    "text":       text[p["BeginOffset"]: p["EndOffset"]],
                    "type":       p["Type"],
                    "type_label": PII_TYPE_LABELS.get(p["Type"], p["Type"]),
                    "score":      round(p["Score"], 4),
                    "begin":      p["BeginOffset"],
                    "end":        p["EndOffset"],
                })

        # 3. Key phrases (all languages)
        kp_resp = client.detect_key_phrases(Text=text, LanguageCode=lang_code)
        for kp in kp_resp.get("KeyPhrases", []):
            result["key_phrases"].append({
                "text":  text[kp["BeginOffset"]: kp["EndOffset"]],
                "score": round(kp["Score"], 4),
                "begin": kp["BeginOffset"],
                "end":   kp["EndOffset"],
            })

    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        error_msg  = e.response["Error"]["Message"]
        raise RuntimeError(f"Comprehend API error [{error_code}]: {error_msg}") from e

    return result
