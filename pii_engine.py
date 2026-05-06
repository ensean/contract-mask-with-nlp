"""
PII Detection and Anonymization Engine

Standard PII:
  - CN_PHONE        中国手机号
  - CN_ID_CARD      居民身份证号
  - CN_BANK_CARD    银行卡号（个人借记/信用卡）
  - EMAIL           电子邮箱
  - CN_ADDRESS      中文地址

Contract entity PII (合同主体信息，默认启用):
  - CN_USCC         统一社会信用代码（18位）
  - CN_BANK_ACCOUNT 对公银行账号（8-20位，区别于个人银行卡）
  - CN_BANK_NAME    开户行名称
  - CN_COMPANY      公司/组织名称（正则兜底 + spaCy ORG）
  - PERSON          自然人姓名（spaCy NER）
"""

import re
from dataclasses import dataclass, field
from typing import Optional

from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_analyzer.nlp_engine import NlpEngineProvider


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

PATTERNS: dict[str, re.Pattern] = {
    # ---- Standard PII ----
    "CN_PHONE": re.compile(
        r"(?<!\d)"
        r"(?:"
        r"(\+?86[-\s]?)?(1[3-9]\d{9})"              # 手机号
        r"|"
        r"(0\d{2,3})[-\s](\d{3,4})[-\s]?(\d{4})"   # 固话：区号[-空格]号码（支持空格分隔）
        r")"
        r"(?!\d)"
    ),
    "CN_ID_CARD": re.compile(
        r"(?<!\d)"
        r"[1-9]\d{5}"
        r"(19|20)\d{2}"
        r"(0[1-9]|1[0-2])"
        r"(0[1-9]|[12]\d|3[01])"
        r"\d{3}[\dXx]"
        r"(?!\d)"
    ),
    "CN_BANK_CARD": re.compile(
        # Matches 16-19 digit card numbers, optionally separated by spaces or hyphens
        # e.g. 6222021234567890123 or 6222 0210 0100 1234 567 or 6222-0210-0100-1234
        r"(?<!\d)"
        r"[3-9]\d{3}"                          # first 4 digits
        r"(?:[\s\-]?\d{4}){3}"                 # 3 groups of 4 (with optional sep)
        r"(?:[\s\-]?\d{1,3})?"                 # optional trailing 1-3 digits
        r"(?!\d)"
    ),
    "EMAIL": re.compile(
        r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"
    ),
    "CN_ADDRESS": re.compile(
        r"[\u4e00-\u9fa5]{2,6}(?:省|自治区|市)"
        r"[\u4e00-\u9fa5]{2,10}(?:市|区|县|镇|乡)"
        r"[\u4e00-\u9fa5\d]{2,30}(?:路|街|道|巷|弄|号|楼|室|单元)[\d\-A-Za-z]*"
        r"(?:[\u4e00-\u9fa5]{0,4}(?:栋|座|幢|号楼|号院)[\d\-A-Za-z\u4e00-\u9fa5]{0,10})?"
        r"(?:[\d\-A-Za-z]{0,6}(?:室|层|楼|单元)[\d\-A-Za-z]{0,6})?"
    ),

    # ---- Contract entity PII ----

    # 统一社会信用代码：1位登记管理部门 + 1位机构类别 + 6位行政区划 + 9位主体标识 + 1位校验
    "CN_USCC": re.compile(
        r"(?<![0-9A-Za-z])"
        r"[0-9A-HJ-NP-RT-Ya-hj-np-rt-y]{2}\d{6}[0-9A-HJ-NP-RT-Ya-hj-np-rt-y]{10}"
        r"(?![0-9A-Za-z])"
    ),

    # 对公银行账号：8-20位数字，支持空格/短横线分隔，需要上下文关键词避免误判
    "CN_BANK_ACCOUNT": re.compile(
        r"(?:账[号户]|卡\s*号|account\s*(?:no\.?|number)?)"
        r"[\s:：]*"
        r"(\d[\d\s\-]{6,22}\d)",               # 8-20 digits with optional spaces/hyphens
        re.IGNORECASE,
    ),

    # 开户行名称：银行名 + 可选地名 + 支行/分行/营业部等
    "CN_BANK_NAME": re.compile(
        r"(?:中国|交通|招商|浦发|光大|华夏|民生|广发|平安|兴业|"
        r"邮储|农业|工商|建设|中信|北京|上海|深圳|广州|"
        r"渤海|恒丰|浙商|徽商|汉口|东莞|宁波|南京|江苏|"
        r"杭州|成都|重庆|西安|郑州|武汉|长沙|天津|青岛)"
        r"[\u4e00-\u9fa5]{0,10}"
        r"(?:银行)"
        r"[\u4e00-\u9fa5]{0,15}"
        r"(?:支行|分行|分公司|营业部|营业所|总行)?"
    ),

    # CN_COMPANY 不用正则，完全依赖 spaCy ORG NER + 后处理过滤
    # （正则误判率极高，已移除）
}


# ---------------------------------------------------------------------------
# Presidio setup
# ---------------------------------------------------------------------------

def _build_analyzer() -> AnalyzerEngine:
    import os
    zh_model = os.getenv("ZH_SPACY_MODEL", "zh_core_web_trf")
    configuration = {
        "nlp_engine_name": "spacy",
        "models": [
            {"lang_code": "zh", "model_name": zh_model},
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
        result = engine.anonymize("甲方：北京科技有限公司，统一社会信用代码：91110000123456789X")
        restored = PIIEngine.restore(llm_response, result.mapping)
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
                # CN_BANK_ACCOUNT pattern uses a capture group for the number only
                if pii_type == "CN_BANK_ACCOUNT" and m.lastindex:
                    # Replace only the account number part, not the keyword prefix
                    start, end = m.span(1)
                    original = m.group(1).strip()
                else:
                    start, end = m.start(), m.end()
                    original = m.group()

                if original.strip():
                    spans.append(PIISpan(
                        start=start, end=end,
                        pii_type=pii_type, original=original,
                    ))
        return spans

    # ------------------------------------------------------------------
    # Presidio-based detection (NER for PERSON + ORG)
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
                    "NRP",
                ],
            )
        except Exception:
            results = []

        # Also run spaCy directly to catch ORG entities (companies)
        # that Presidio doesn't expose by default
        org_spans = self._spacy_org_spans(text, language)
        # FAC/LOC spans supplement CN_ADDRESS regex for building/facility names
        addr_spans = self._spacy_address_spans(text, language)

        # Non-name words that spaCy/Presidio sometimes misclassify as PERSON
        _NON_PERSON_WORDS = frozenset([
            "张贴", "送达", "签收", "披露", "接收", "甲方", "乙方", "双方",
            "第三方", "当事人", "代理人", "委托", "授权", "法院", "仲裁",
            "通知", "文书", "留置", "邮寄", "退回",
        ])

        spans: list[PIISpan] = []
        for r in results:
            text_val = text[r.start:r.end]
            if r.entity_type == "PERSON":
                # Skip if contains known non-person words
                if any(w in text_val for w in _NON_PERSON_WORDS):
                    continue
                # Chinese name should be 2-8 chars
                cn_chars = sum(1 for c in text_val if "\u4e00" <= c <= "\u9fa5")
                if cn_chars > 0 and not (2 <= cn_chars <= 8):
                    continue
            spans.append(PIISpan(
                start=r.start, end=r.end,
                pii_type=r.entity_type,
                original=text_val,
            ))
        return spans + org_spans + addr_spans

    def _spacy_org_spans(self, text: str, language: str) -> list[PIISpan]:
        """
        Extract company/organization names via spaCy NER.

        Two problems with zh_core_web_sm we work around:
        1. "北京星辰科技有限公司" is split into GPE("北京") + ORG("科技有限公司"),
           with non-entity text "星辰" in between.
           We look ahead up to 10 chars after a GPE to find an adjacent ORG and
           merge the whole span (GPE text + gap + ORG text) into one CN_COMPANY.
        2. Generic ORG entities like "仲裁委员会" or "人民法院" are NOT company
           principals — we filter by requiring at least one company-suffix keyword.
        """
        COMPANY_SUFFIXES = frozenset([
            "公司", "企业", "集团", "控股", "股份", "有限", "合伙",
            "事务所", "基金", "银行", "保险", "证券", "投资", "实业",
            "科技", "传媒", "文化", "网络", "信息", "技术", "工程",
            "建设", "贸易", "咨询", "服务", "发展", "电子", "医疗",
            "教育", "金融", "商贸", "物流", "能源", "地产", "置业",
        ])
        # Prefixes that indicate the text is NOT a company name
        NON_COMPANY_PREFIXES = frozenset([
            "甲方", "乙方", "双方", "第三方", "对方", "一方", "任何",
            "收款", "付款", "开户", "账户",
        ])
        # Max gap (chars) between a GPE and a following ORG to merge them
        MAX_MERGE_GAP = 10

        try:
            import os, spacy
            default_zh = "zh_core_web_trf"
            model = os.getenv("ZH_SPACY_MODEL", default_zh) if language == "zh" else "en_core_web_sm"
            nlp = spacy.load(model)
            doc = nlp(text)
            ents = list(doc.ents)
            spans: list[PIISpan] = []
            skip_next = False
            for i, ent in enumerate(ents):
                if skip_next:
                    skip_next = False
                    continue

                # Try to merge GPE + (gap ≤ MAX_MERGE_GAP chars) + ORG
                if ent.label_ == "GPE" and i + 1 < len(ents):
                    next_ent = ents[i + 1]
                    gap = next_ent.start_char - ent.end_char
                    if next_ent.label_ in ("ORG", "COMPANY") and 0 <= gap <= MAX_MERGE_GAP:
                        merged_text = text[ent.start_char: next_ent.end_char]
                        if (any(kw in merged_text for kw in COMPANY_SUFFIXES)
                            and not any(merged_text.startswith(p) for p in NON_COMPANY_PREFIXES)):
                            spans.append(PIISpan(
                                start=ent.start_char,
                                end=next_ent.end_char,
                                pii_type="CN_COMPANY",
                                original=merged_text,
                            ))
                            skip_next = True
                            continue

                if ent.label_ in ("ORG", "COMPANY"):
                    if (any(kw in ent.text for kw in COMPANY_SUFFIXES)
                            and not any(ent.text.startswith(p) for p in NON_COMPANY_PREFIXES)):
                        spans.append(PIISpan(
                            start=ent.start_char, end=ent.end_char,
                            pii_type="CN_COMPANY",
                            original=ent.text,
                        ))
            return spans
        except Exception:
            return []

    def _spacy_address_spans(self, text: str, language: str) -> list[PIISpan]:
        """
        Extract FAC (facility/building) and LOC (location) entities via spaCy
        as CN_ADDRESS supplements.

        FAC catches building-level details like "碧波路690号3号楼" that the
        CN_ADDRESS regex may miss when the full province/city prefix is absent.
        LOC catches sub-district locations like "朝阳区".

        Only applies to Chinese text; English addresses are handled by Presidio.
        Minimum length filter (≥4 chars) avoids single-char noise.
        """
        if language != "zh":
            return []

        # Tokens that look like FAC/LOC but are not address-sensitive
        _NON_ADDRESS = frozenset([
            "中华人民共和国", "中国", "全国", "境内", "境外", "海外",
        ])

        try:
            import os, spacy
            model = os.getenv("ZH_SPACY_MODEL", "zh_core_web_trf")
            nlp = spacy.load(model)
            doc = nlp(text)
            spans: list[PIISpan] = []
            for ent in doc.ents:
                if ent.label_ not in ("FAC", "LOC"):
                    continue
                val = ent.text.strip()
                if len(val) < 4:          # skip short noise
                    continue
                if val in _NON_ADDRESS:   # skip country-level non-sensitive
                    continue
                spans.append(PIISpan(
                    start=ent.start_char, end=ent.end_char,
                    pii_type="CN_ADDRESS",
                    original=val,
                ))
            return spans
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Merge & deduplicate overlapping spans
    # Priority: regex > presidio/spacy NER
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_spans(
        dict_spans: list[PIISpan],
        regex_spans: list[PIISpan],
        presidio_spans: list[PIISpan],
    ) -> list[PIISpan]:
        """
        Merge three span sources with priority: dict > regex > presidio/NER.
        Higher-priority spans block lower-priority spans that overlap with them.
        """
        def remove_overlapping(candidates: list[PIISpan], blockers: list[PIISpan]) -> list[PIISpan]:
            blocker_ranges = [(s.start, s.end) for s in blockers]
            return [
                s for s in candidates
                if not any(s.start < be and s.end > bs for bs, be in blocker_ranges)
            ]

        filtered_regex    = remove_overlapping(regex_spans, dict_spans)
        filtered_presidio = remove_overlapping(presidio_spans, dict_spans + filtered_regex)
        all_spans = dict_spans + filtered_regex + filtered_presidio

        if not all_spans:
            return []

        sorted_spans = sorted(all_spans, key=lambda s: (s.start, -(s.end - s.start)))
        merged: list[PIISpan] = [sorted_spans[0]]
        for span in sorted_spans[1:]:
            if span.start < merged[-1].end:
                continue
            merged.append(span)
        return merged

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def anonymize(self, text: str, language: str = "zh") -> AnonymizationResult:
        """Replace PII in *text* with placeholders. Resets counter before each call."""
        self._reset_counter()
        return self._anonymize_no_reset(text, language)

    def _anonymize_no_reset(self, text: str, language: str = "zh") -> AnonymizationResult:
        """Like anonymize() but does NOT reset the counter.
        Use when processing multiple texts sharing one document-level counter."""
        # Priority: dict > regex > presidio/spacy NER
        from dict_engine import get_dict
        dict_hits = get_dict().find_hits(text)
        dict_spans = [
            PIISpan(start=h.start, end=h.end, pii_type=h.group, original=h.term)
            for h in dict_hits
        ]

        regex_spans = self._regex_spans(text)
        presidio_spans = self._presidio_spans(text, language)
        all_spans = self._merge_spans(dict_spans, regex_spans, presidio_spans)

        mapping: dict[str, str] = {}
        for span in all_spans:
            placeholder = self._next_placeholder(span.pii_type)
            span.placeholder = placeholder
            mapping[placeholder] = span.original

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
