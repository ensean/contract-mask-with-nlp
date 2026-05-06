# PII-Safe LLM Chat

基于 FastAPI + AWS Bedrock 的合同 PII 脱敏系统。支持文本对话和 Word 文档处理。

## 功能

| 功能 | 说明 |
|------|------|
| 💬 文本对话 | 输入文本自动脱敏后发给大模型，回复还原 PII 后展示 |
| 📄 Word 脱敏 | 上传 `.docx`，输出带占位符的脱敏文档 + Session ID |
| 🔓 Word 还原 | 上传脱敏文档 + Session ID，还原为原始文档 |
| 🔬 Comprehend 分析 | 用 AWS Comprehend 识别实体、关键短语（中文支持） |
| 📋 敏感词典 | 在线编辑 `sensitive_dict.txt`，热更新，无需重启 |

## PII 识别范围

**通用 PII（正则）**
- `CN_PHONE` 中国手机号
- `CN_ID_CARD` 居民身份证号
- `CN_BANK_CARD` 银行卡号（个人）
- `EMAIL` 电子邮箱
- `CN_ADDRESS` 中文地址

**合同主体信息（正则 + NER）**
- `CN_COMPANY` 公司/组织名称（spaCy `zh_core_web_trf` ORG）
- `CN_USCC` 统一社会信用代码（18位）
- `CN_BANK_ACCOUNT` 对公银行账号（上下文锚定）
- `CN_BANK_NAME` 开户行名称
- `PERSON` 自然人姓名（spaCy `zh_core_web_trf` PERSON）

**合同字段提取（标签锚定正则，最高优先级）**
- `名称：`、`法定代表人：`、`联系人：`、`住所：` 等字段后的值

**词典匹配（`sensitive_dict.txt`，优先级高于 NER）**
- 支持 `[GROUP_NAME]` 分组，占位符格式 `<<GROUP_NAME_N>>`
- 精确匹配，英文忽略大小写，热更新

**识别优先级：** 词典 > 合同字段正则 > 通用正则 > spaCy NER

## 快速启动

```bash
# 安装依赖
pip install -r requirements.txt
python -m spacy download zh_core_web_trf
python -m spacy download en_core_web_sm

# 启动服务
.venv/bin/uvicorn main:app --reload --port 8000
```

浏览器访问 http://localhost:8000

## AWS 配置

使用 `test` AWS profile，需要有 Bedrock 和 Comprehend 访问权限：

```bash
aws configure --profile test
```

**支持的大模型：**

| Key | 模型 | Context |
|-----|------|---------|
| `claude-sonnet-4-6` | Claude Sonnet 4.6（默认） | 1M tokens |
| `kimi-k2.5` | Kimi K2.5 (Moonshot AI) | 256K tokens |
| `glm-5` | GLM 5 (Z.AI) | 200K tokens |
| `minimax-m2.5` | MiniMax M2.5 | 196K tokens |

区域：`us-east-1`

## 项目结构

```
├── main.py                # FastAPI 应用入口
├── pii_engine.py          # PII 识别与脱敏引擎
├── bedrock_client.py      # AWS Bedrock 调用封装
├── comprehend_client.py   # AWS Comprehend 调用封装
├── word_processor.py      # Word 文档脱敏/还原
├── dict_engine.py         # 词典匹配引擎（热更新）
├── sensitive_dict.txt     # 敏感词典（可在线编辑）
├── templates/
│   └── index.html         # 前端页面（5个Tab）
├── sessions/              # 脱敏 Session 映射存储
├── uploads/               # 临时文件目录
└── requirements.txt
```

## 内网部署

`zh_core_web_trf` 模型（~400MB BERT）安装在：
```
.venv/lib/python3.12/site-packages/zh_core_web_trf/zh_core_web_trf-3.8.0/
```

内网环境可通过环境变量指定本地模型路径（无需联网）：
```bash
export ZH_SPACY_MODEL=/opt/models/zh_core_web_trf-3.8.0
```

或直接打包 `.venv` 目录传输到内网机器。
