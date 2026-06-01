# PII-Safe LLM Chat

基于 FastAPI + AWS Bedrock 的合同 PII 脱敏系统。支持文本对话和 Word 文档处理。

## 功能

| 功能 | 说明 |
|------|------|
| 💬 文本对话 | 输入文本自动脱敏后发给大模型，回复还原 PII 后展示 |
| 📄 Word 脱敏 | 上传 `.docx`，输出带占位符的脱敏文档 + Session ID |
| 🔓 Word 还原 | 上传脱敏文档 + Session ID，还原为原始文档 |
| 🤖 一条龙审阅 | 上传合同 → 脱敏 → 大模型审阅（仅发送脱敏文本）→ **Word 批注**给出建议 → 还原后下载，同时提供脱敏证据副本 |
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

## Docker 部署（app + Redis）

提供了 `Dockerfile` 与 `docker-compose.yml`，一条命令拉起服务 + Redis：

```bash
docker compose up -d --build
```

- `app`：构建时已把 spaCy 模型（含 ~400MB 的 `zh_core_web_trf`）打进镜像，容器离线即可启动。
- `redis`：开启 AOF 持久化，作为异步任务状态的共享存储（`REDIS_URL` 已在 compose 中配好）。
- **AWS 凭证**：compose 把宿主机 `~/.aws` 以**只读**方式挂进容器，并设 `AWS_PROFILE=test`，
  boto3 直接复用本机的 `test` profile，无需把密钥写进镜像。
- **数据持久化**：`sessions/`（占位符→PII 明文映射）挂载到宿主机目录保留；
  Redis 数据存命名卷 `redis-data`；`uploads/` 是临时文件，不持久化。

**扩多 worker**：因为任务状态已走 Redis，可安全提高并发——在 compose 的 `app` 服务下取消注释：

```yaml
    command: uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

> 首次 `--build` 会下载 torch 和 BERT 模型，镜像较大、耗时较久属正常。

## AWS 配置（本地直接运行时）

使用 `test` AWS profile，需要有 Bedrock 和 Comprehend 访问权限：

```bash
aws configure --profile test
```

**支持的大模型：**

| Key | 模型 | Context |
|-----|------|---------|
| `claude-sonnet-4-6` | Claude Sonnet 4.6（默认） | 1M tokens |
| `claude-opus-4-8` | Claude Opus 4.8 | 1M tokens |

区域：Bedrock 用 `ap-northeast-1`，Comprehend 用 `us-east-1`。

> 注：Claude Opus 4.7+ 弃用了 `temperature` 等采样参数，代码对这类模型会自动省略该参数。

## 一条龙合同审阅流程

`🤖 一条龙审阅` Tab（`POST /docx/review`）将脱敏、大模型审阅、批注、还原串成一步：

```
上传 .docx
  → ① 脱敏：在内存中的文档对象上把 PII 替换为占位符，建立 session 映射
  → ② 从脱敏后的文档提取【纯文本】，发给 Bedrock 审阅
       —— 离开本地的只有脱敏文本字符串，docx 文件本身从不外发
  → ③ 还原文档（占位符 → 原始 PII，run 级替换以保留批注锚点）
  → ④ 将建议作为 Word 批注（w:comment）锚定到还原后的原文
  → ⑤ 返回「带批注的还原文档」+「脱敏证据副本」供下载
```

**关于"发给大模型的内容"——重要：**
发给 LLM 的是 `extract_document_text(doc)` 提取的**脱敏后纯文本字符串**
（Converse API 的 `messages[].content[].text`），**不是 docx 文件**。
无论原始 / 脱敏 / 还原态的 docx 文件，全程都留在服务端，绝不上传给模型。

**LLM 输出格式（分隔符，非 JSON）：**
模型按如下分隔符块返回，避免中文内容里的引号 / 换行破坏解析（JSON 在此场景极易损坏）：

```
@@QUOTE@@
<合同原文中需逐字匹配的定位片段>
@@COMMENT@@
<审阅意见，可多行、可含任意标点引号>
@@END@@
```

无问题时模型只回一行 `@@NONE@@`。

关键设计：

- **PII 不出脱敏边界** —— 大模型只看到占位符纯文本；审阅意见中的占位符在写入批注前才还原。
- **占位符文档级去重** —— 同一原始值（如同一公司名）在全文使用**同一个**占位符编号，
  避免模型因 `<<CN_COMPANY_1>>` / `<<CN_COMPANY_4>>` 不同而误判为不同主体。
- **批注精确锚定** —— 在批注片段边界处拆分 run（`mark_comment_range`），将批注绑定到精确文本范围而非整段，同时保留原有格式。
- **跨段 / 定位失败兜底** —— 若模型给出的 `quote` 跨段落或无法逐字匹配，自动按行 / 句拆分，
  用最长可匹配子片段锚定；仍无法定位的列入 `comments_unmatched` 在前端提示，不影响其余批注。
- **脱敏证据副本** —— 同时返回脱敏版 docx 供下载，用于核对原始 PII 从未外发。
- 依赖 `python-docx>=1.2.0` 的原生批注 API（`Document.add_comment`）。

**异步任务模式（应对 CloudFront / 代理超时）：**
审阅要调大模型，单请求耗时 10-40s，会超出 CloudFront 默认 30s（最高约 60s）的 origin 响应超时。
因此 `/docx/review` 改为「提交 + 轮询」两段式，每个请求都是毫秒级，不触发任何代理/CDN 超时：

```
POST /docx/review            上传文档，立即返回 { job_id, status: "processing" }
                             重活在后台线程池跑（job_manager.py）
GET  /docx/review/{job_id}   轮询：
                               processing → { status, elapsed }
                               done       → { status, elapsed, result: {...} }
                               error      → 502 { status, detail }
```

- 前端提交后每 3s 轮询一次，显示「审阅中…（已等待 Ns）」，完成后再渲染结果与下载。
- 任务状态默认存于进程内存（`MemoryJobStore`），适用于**单 worker** 部署；完成的任务保留 30 分钟（TTL）后回收。
- **多 worker / 多实例部署**：设置环境变量 `REDIS_URL`（如 `redis://host:6379/0`），任务状态改存 Redis（`RedisJobStore`），任何 worker 都能响应轮询，避免轮询被路由到别的 worker 而 404。未设置或 Redis 不可达时自动回退到内存存储。

  ```bash
  export REDIS_URL=redis://localhost:6379/0   # 可选，多 worker 时需要
  export JOB_TTL_SECONDS=1800                 # 任务保留时长（默认 1800）
  export JOB_MAX_WORKERS=2                    # 后台线程池大小（默认 2）
  ```

### 性能与并发说明

- **spaCy 模型缓存**：`zh_core_web_trf`（~400MB BERT）按模型名缓存为进程级单例，仅首次加载；
  之前每段落每文档都 `spacy.load()` 重载，是审阅慢和并发内存膨胀的主因。修复后单次脱敏从数秒降到毫秒级。
- **NER 线程安全**：spaCy 的 `nlp()` 对同一管线并发调用不保证线程安全，故按模型加锁串行化推理；
  审阅在线程池运行，多文档并发时 NER 步骤会串行，但各自结果完全隔离，不会串档。
- **会话隔离**：每个请求的临时文件、`job_id`、`session_id` 均为独立 UUID；一条龙流程内部每次新建独立
  `PIIEngine` 实例，占位符计数器互不干扰。多人同时提交不会发生 PII 串档。

## 文件存储与生命周期

| 类型 | 位置 | 生命周期 |
|------|------|----------|
| 上传 / 中间态 / 最终态 docx | `uploads/{uid}_*.docx` | **临时落盘**，读成 base64 回传后在 `finally` 中删除（`/docx/restore` 的 `_restored.docx` 例外，由 `FileResponse` 直接返回） |
| 脱敏映射 | `sessions/{session_id}.json` | **持久化**，内容为 `占位符 → 原始 PII` 的明文，保留至手动清理 |

- 文档通过 base64 直接回传浏览器生成下载，服务端不长期保留文件。
- `sessions/*.json` 是明文敏感数据；`sessions/` 与 `uploads/` 已加入 `.gitignore`。
- 路径为相对工作目录的 `uploads/`、`sessions/`，即从启动 uvicorn 的目录算起。

## 项目结构

```
├── main.py                # FastAPI 应用入口
├── pii_engine.py          # PII 识别与脱敏引擎
├── bedrock_client.py      # AWS Bedrock 调用封装
├── comprehend_client.py   # AWS Comprehend 调用封装
├── word_processor.py      # Word 文档脱敏/还原 + 一条龙审阅批注
├── job_manager.py         # 异步任务管理（内存 / Redis 双后端，按 REDIS_URL 自动选择）
├── dict_engine.py         # 词典匹配引擎（热更新）
├── sensitive_dict.txt     # 敏感词典（可在线编辑）
├── Dockerfile             # 应用镜像（spaCy 模型已内置）
├── docker-compose.yml     # app + Redis 一键部署
├── templates/
│   └── index.html         # 前端页面（6个Tab）
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

## 开发工具

本项目使用 [Kiro](https://kiro.dev) 开发 —— 一款由 AWS 出品的 AI 驱动开发环境，支持 Spec 驱动开发、智能代码补全和 Agent 自动化工作流。
