# PII-Safe LLM Chat

基于 FastAPI + AWS Bedrock (Claude) 的 PII 脱敏聊天系统。

## 功能

- 用户输入 → **PII 自动识别并替换为占位符** → 发送给大模型
- 大模型返回 → **占位符自动还原为原始 PII** → 展示给用户
- 支持识别：中文姓名、手机号、身份证号、银行卡号、邮箱、地址，以及英文 PERSON/EMAIL/PHONE 等

## 快速启动

```bash
# 启动服务
.venv/bin/uvicorn main:app --reload --port 8000
```

浏览器访问 http://localhost:8000

## 项目结构

```
├── main.py           # FastAPI 应用入口
├── pii_engine.py     # PII 识别与脱敏引擎
├── bedrock_client.py # AWS Bedrock 调用封装
├── templates/
│   └── index.html    # 前端页面
└── requirements.txt
```

## AWS 配置

使用 `test` AWS profile，需要有 Bedrock 访问权限：

```bash
aws configure --profile test
```

默认模型：`anthropic.claude-3-5-sonnet-20241022-v2:0`，区域：`us-east-1`
