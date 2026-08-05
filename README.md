# QQ Bot — Simplified / Greenfield

按 `QQBOT_SIMPLIFIED_DESIGN.md` 从零重构的简化版 QQ 群机器人。

分支策略：

- `refactor/simplified`（当前分支）：完全重写的 `qqbot/` 包，是本次重构产物。
- `main`：保留作参考，未做改动；新实现不复用其代码。

## 模块布局（对应设计文档章节）

| 模块 | 职责 | 设计章节 |
|---|---|---|
| `qqbot/ingress.py` | OneBot v11 反向 WebSocket 接入（NapCat） | §15 |
| `qqbot/aggregate.py` | 3 秒聚合窗口（多段消息合并） | §15 |
| `qqbot/prefilter.py` | 预过滤：白名单/撤回/过期/过短/无 @ | §10 |
| `qqbot/gate.py` | 低成本候选门控：纯程序判定是否值得调用 LLM | §4 |
| `qqbot/classifier.py` | 单一语义分类（`skip/chat/knowledge`） | §3.2 |
| `qqbot/context.py` | 上下文构建（闲聊/知识/追问三套）+ 追问扩展打分 | §6 |
| `qqbot/llm.py` | 统一异步 LLM 调用（JSON 容错 + 重试 + 可注入 fake） | §7 |
| `qqbot/embedding.py` | 哈希 n-gram 词法兜底 + 可选 API 语义 | §8.3 |
| `qqbot/knowledge.py` | Markdown 按标题切 chunk + BM25 词法 + 向量混合检索 + 重排 | §8 |
| `qqbot/generate.py` | 闲聊 / 知识生成（§7.2 / §7.3，无复审 LLM） | §7 |
| `qqbot/safety.py` | 数值 grounding / 长度 / 去重 / 身份保护 | §11 |
| `qqbot/send.py` | 限流 / 群发送锁 / 过期丢弃 / dry_run | §12 |
| `qqbot/audit.py` | 审计（每步留痕，可离线回放） | §13 |
| `qqbot/store.py` `qqbot/memory.py` | SQLite 持久化 + 长期记忆 worker | §9 |
| `qqbot/queue.py` | 内存队列 + 持久化 journal | §14 |
| `qqbot/server.py` `run.py` | 主链路编排 / 入口 | §15/§18 |

## 运行

```bash
cp .env.example .env          # 填写 LLM_API_KEY / BOT_QQ / WHITELISTED_GROUPS
DRY_RUN=1 python run.py       # 仅审计不发送，先验证链路
python run.py                 # 正式运行（需 NapCat 反向 WS 连到 ONE_BOT_WS_PORT）
```

## 测试

```bash
pytest tests/                 # 预过滤 / 候选门控 / 知识检索 / 安全校验 / 全链路 dry_run
```

## 当前状态

旧 `main` 运行实例的进程已停止，其 LaunchAgent 自动运行已禁用。本分支尚未部署上线；部署前需：填好 `.env`、确认 embedding 后端、按 §19 调参（上下文预算 / 记忆召回 / 限流参数）。
