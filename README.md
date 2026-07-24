# Squad QQBot MVP

这是一个本地运行的《Squad / 战术小队》新兵问答 QQBot。

它不是通用聊天机器人，而是挂在 QQ 群里的“Squad 新兵营教官”：只在白名单群里观察消息，尽量只回答玩法、兵种、服务器、语音软件和故障排查相关的问题。

## 当前功能

- 接收 NapCat / OneBot HTTP 上报的 QQ 群消息。
- 只处理 `ALLOWED_GROUP_IDS` 配置的白名单群。
- 支持被 `@机器人` 后优先回答。
- 白名单群普通消息也会观察，但必须像 Squad 相关问题或求助才会主动回复。
- 使用本地 Markdown 知识库检索上下文。
- 使用检索分数和查询词覆盖率区分强命中、弱命中，避免拿弱相关资料硬答。
- 被 @ 的问题未命中知识库时，由独立的通用模型提示词谨慎兜底。
- 可在白名单群里低频参与闲聊，使用可配置的近期消息和回复关系作为上下文。
- 按群异步维护滚动场景快照，提炼当前话题、指代关系和接话位置，不阻塞当前回复。
- @/管理消息、普通问答和闲聊分别由独立 worker 处理，闲聊不会阻塞 @ 消息。
- 调用 DeepSeek / MiMo 等 OpenAI 兼容接口生成回答。
- 回答不展示参考资料，不使用 Markdown 格式。
- 发送前统一清理 Markdown、限制为 500 字，并强制把 Rally/Rally Point 改成“队包”。
- 每分钟最多回复 8 条，避免突发刷屏。
- 闲聊回复按群限制为至少间隔 60 秒、每小时最多 20 条。
- 普通消息超过 60 秒、@ 消息超过 5 分钟后不再回复。
- 同一用户连续发送的消息会等待 3 秒并按说话对象聚合，最长等待 8 秒；QQ 的 @/回复关系作为硬目标，目标不明的后续片段由语义分类器判断，低置信度时不继承对机器人的 @。
- 同一群同一知识主题默认 60 秒内不重复主动回复。
- 同一群同一用户的短追问默认保留 120 秒上下文。
- 同群不同用户接话默认保留 30 秒上下文。
- 被 @ 的追问默认保留 180 秒上下文。
- QQ 显式回复机器人时按机器人消息 ID 精确加载原问题、原回答和知识来源；对话轮次写入 SQLite，重启后仍可恢复。
- 没有显式回复链时，只在消息含有明确承接词且话题不冲突时关联上一轮，不再把所有短句都当成追问。
- 知识回答会用近期群聊解析“那个、这个”等指代；首次检索较弱时才语义改写独立检索问题，群聊不直接作为事实依据。
- 支持本地 `/ask` 调试接口。
- 支持 `/health` 健康检查。
- 支持群内 `重载知识库` 或 `reload` 重载知识库。
- 支持管理员维护命令：健康状态、最近跳过、开关自动回复。
- 记录群消息处理审计日志，方便后续排查漏回、误回和知识库空缺。
- 使用 SQLite 持久化待处理消息；普通消息只恢复 60 秒内的，@ 消息只恢复 5 分钟内的，管理命令不重放。
- 非白名单群普通消息不写审计日志；只有非白名单群里 @ 机器人时才记录，避免刷屏。

## 目录结构

```text
squad-qqbot-mvp/
  squad_bot/
    config.py       配置
    knowledge.py    Markdown 知识库检索
    llm.py          大模型调用和回复判断
    onebot.py       OneBot 消息处理
    server.py       HTTP 服务和群消息队列
  knowledge/        Squad 知识库
  scripts/
    com.squad.qqbot.mvp.plist   launchd 配置
    sync_runtime.sh             同步源码到运行副本
    rotate_logs.sh              日志轮转
    replay_chat_audit.py        回放审计日志中的闲聊场景
  tests/            单元测试
  work/             运行调试文件和消息审计日志
  .env.example      配置模板
  run.py            启动入口
  test_local.py     本地测试脚本
```

## 配置

复制配置模板：

```bash
cp .env.example .env
```

然后编辑 `.env`：

```text
BOT_HOST=127.0.0.1
BOT_PORT=8088
BOT_QQ=你的机器人QQ号
ADMIN_QQ_IDS=管理员QQ号1,管理员QQ号2
ALLOWED_GROUP_IDS=群号1,群号2

ONEBOT_API_URL=http://127.0.0.1:3000
ONEBOT_ACCESS_TOKEN=

LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-chat
LLM_API_KEY=你的DeepSeek或MiMoKey
CHAT_MODEL=

KNOWLEDGE_DIR=knowledge
MAX_CONTEXT_CHARS=4500
KNOWLEDGE_STRONG_MIN_SCORE=0.18
KNOWLEDGE_STRONG_MIN_COVERAGE=0.6
MAX_ANSWER_CHARS=500
MAX_REPLIES_PER_MINUTE=8
NORMAL_MESSAGE_MAX_AGE_SECONDS=60
MENTIONED_MESSAGE_MAX_AGE_SECONDS=300
NORMAL_REPLY_DELAY_SECONDS=0
MESSAGE_FRAGMENT_DEBOUNCE_SECONDS=3
MESSAGE_FRAGMENT_MAX_WAIT_SECONDS=8
MESSAGE_FRAGMENT_MAX_PARTS=6
MESSAGE_FRAGMENT_MAX_CHARS=800
MESSAGE_FRAGMENT_SEMANTIC_ENABLED=true
MESSAGE_FRAGMENT_SEMANTIC_MODEL=
MESSAGE_FRAGMENT_SEMANTIC_TIMEOUT_SECONDS=8
MESSAGE_FRAGMENT_SEMANTIC_MIN_CONFIDENCE=0.75
CONTEXTUAL_QUERY_ENABLED=true
CONTEXTUAL_QUERY_MODEL=
CONTEXTUAL_QUERY_TIMEOUT_SECONDS=8
CONTEXTUAL_QUERY_MIN_CONFIDENCE=0.75
SAME_TOPIC_COOLDOWN_SECONDS=60
FOLLOWUP_SAME_USER_SECONDS=120
FOLLOWUP_GROUP_SECONDS=30
FOLLOWUP_MENTION_SECONDS=180
MESSAGE_AUDIT_LOG=work/message_audit.jsonl
PENDING_QUEUE_DB=work/pending_queue.sqlite3
AUTO_REPLY_ENABLED=true
LLM_FALLBACK_ENABLED=true
FALLBACK_ONLY_WHEN_MENTIONED=true
CHAT_REPLY_ENABLED=true
CHAT_ALLOWED_GROUP_IDS=允许闲聊的群号1,群号2
CHAT_REPLY_COOLDOWN_SECONDS=60
MAX_CHAT_REPLIES_PER_HOUR=20
CHAT_CONTEXT_SECONDS=300
CHAT_CONTEXT_MESSAGES=12
CHAT_REPLY_DEBOUNCE_SECONDS=2
CHAT_SCENE_ENABLED=true
CHAT_SCENE_DEBOUNCE_SECONDS=3
CHAT_SCENE_UPDATE_INTERVAL_SECONDS=30
CHAT_SCENE_STALE_SECONDS=600
CHAT_SCENE_MIN_MESSAGES=3
CHAT_SCENE_TIMEOUT_SECONDS=30
# CHAT_SCENE_MODEL=可选；不配置时复用 LLM_MODEL
DRY_RUN=false
```

如果使用 MiMo Token Plan，通常这样配置：

```text
LLM_BASE_URL=https://token-plan-cn.xiaomimimo.com/v1
LLM_MODEL=mimo-v2.5-pro
CHAT_MODEL=mimo-v2.5
```

`CHAT_MODEL` 只控制最终闲聊回复，留空时复用 `LLM_MODEL`。上面的 MiMo 配置会让知识问答、追问改写和场景分析继续使用 `mimo-v2.5-pro`，最终闲聊使用 `mimo-v2.5`。

不要把 `.env` 里的真实 API Key 发到聊天、截图或公开仓库里。

## 启动

```bash
PYTHONPYCACHEPREFIX=/private/tmp/codex_pycache python3 run.py
```

启动后默认监听：

```text
http://127.0.0.1:8088
```

健康检查：

```bash
curl http://127.0.0.1:8088/health
```

正常结果类似：

```json
{"ok": true, "chunks": 199}
```

修改 Python 代码后需要重启服务。重启前先停止当前 `run.py` 进程，避免 8088 端口占用。

当前推荐使用 launchd 常驻运行，配置在 `scripts/com.squad.qqbot.mvp.plist`。由于 macOS 可能限制系统服务读取 `Documents` 目录，常驻服务实际从 `/Users/yuce/.codex/qqbot-runtime/squad-qqbot-mvp` 运行。修改源码、配置或知识库后，先执行 `sh scripts/sync_runtime.sh` 同步运行副本，再重启 launchd 服务。

## 本地测试

先启动服务，再请求：

```bash
curl -X POST http://127.0.0.1:8088/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"HAB 和 FOB 有什么区别？"}'
```

也可以运行：

```bash
python3 test_local.py
```

## 回放审计日志

`scripts/replay_chat_audit.py` 可以回放审计日志中的闲聊场景，用于调试提示词效果，不会发送任何 QQ 消息。

```bash
# 回放最近 5 条闲聊记录
python3 scripts/replay_chat_audit.py

# 回放包含特定关键词的记录
python3 scripts/replay_chat_audit.py --contains "试卷" --limit 10

# 指定模型回放（对比不同模型效果）
python3 scripts/replay_chat_audit.py --model mimo-v2.5 --limit 10

# 输出到文件
python3 scripts/replay_chat_audit.py --limit 20 --output replay_result.jsonl
```

输出每行是一条 JSON，包含原问题、原回答、回放回答和延迟，方便对比优化效果。

## 接入 QQ

推荐路线：

- NapCat 负责登录 QQ，并提供 OneBot HTTP API。
- 本项目作为问答服务，接收 OneBot 的 HTTP 上报。

OneBot 侧需要配置：

```text
HTTP 上报地址：http://127.0.0.1:8088/onebot
API 地址：填到本项目 .env 的 ONEBOT_API_URL
```

群里可以直接问 Squad 相关问题，也可以 @ 机器人：

```text
@机器人 HAB 和 FOB 有啥区别
```

普通群消息不会因为提到关键词就必然回复。知识问答、未命中兜底和闲聊使用三套独立提示词。闲聊只在 `CHAT_ALLOWED_GROUP_IDS` 中低频触发，并过滤公告、链接、资料整理、任务安排、bot 元讨论和群友之间的定向对话。发给模型的近期群聊会把真实 QQ 号匿名化为“群友A、群友B”。场景快照在后台按群合并更新；生成回复时会同时看到快照和原始上下文，并以当前消息、引用关系和原始上下文为准。

## 消息审计日志

运行中会写入：

```text
work/message_audit.jsonl
```

每行是一条 JSON，包含：

- 时间、群号、用户 QQ。
- 原始问题文本。
- 是否 @ 机器人。
- 处理结果：`answered`、`answered_dry_run`、`skipped`、`ignored`。
- 跳过或回答原因。
- 是否命中知识库上下文。
- 命中的知识库来源标题。

这个文件用于定期复盘：哪些问题没答上、哪些消息被过滤、哪些知识点需要补进知识库。

为了避免日志刷屏，非白名单群的普通消息不会写入审计日志；非白名单群里 @ 机器人时会记录为 `group not allowed`。

## 更新知识库

知识库都在 `knowledge/` 目录里，直接新建或编辑 Markdown 文件即可。

建议每篇文章写清楚：

```markdown
# 标题

来源：
- 链接或说明

## 问题标题

回答内容
```

修改 Markdown 知识库后，管理员可以在群里发：

```text
重载知识库
```

或：

```text
reload
```

修改 Python 代码后必须重启服务。

## 管理命令

管理命令只允许群主、群管理员，或 `ADMIN_QQ_IDS` 里的 QQ 执行。管理员可以在群里发：

```text
重载知识库
健康状态
最近跳过
关闭自动回复
开启自动回复
```

也兼容：

```text
/问 重载知识库
/问 health
/问 auto off
/问 auto on
```

关闭自动回复后，普通群消息不会主动回复；被 @ 的知识库问题仍可回答。

## 已知边界

- 只有可配置时间窗内的短期闲聊上下文和内存场景快照，没有长期聊天记忆；重启后快照会随新消息重新建立。
- 闲聊回复时间写入 SQLite，因此重启服务不会重置 60 秒冷却和每小时上限。
- 主动回复策略仍需要根据真实群日志继续调优。
- 当前是关键词和短语检索，还没有向量检索。
- 还没有 ST 私有知识库，涉及本服规则时应以群公告和管理员为准。
- 当前使用本机 launchd 常驻运行；源码修改后仍需先同步运行副本再重启服务。

## 后续建议

- 根据审计日志继续校准强命中阈值和闲聊接话风格。
- 后续可升级 SQLite FTS、embedding 或向量库检索。

## 公开资料参考

- Squad Wiki: https://squad.fandom.com/wiki/Squad_Wiki
- Forward Operating Base: https://squad.fandom.com/wiki/Forward_Operating_Base
- 队包: https://squad.fandom.com/wiki/Rally_Point
- Logistic System: https://squad.fandom.com/wiki/Logistic_System
- Squad Leader: https://squad.fandom.com/wiki/Squad_Leader
- Medic: https://squad.fandom.com/wiki/Medic
- Rifleman: https://squad.fandom.com/wiki/Rifleman
- Light Anti-Tank: https://squad.fandom.com/wiki/Light_Anti-Tank
- Heavy Anti-Tank: https://squad.fandom.com/wiki/Heavy_Anti-Tank

B 站和小黑盒内容建议优先使用你认可的具体帖子或视频链接，再逐条补进知识库，避免把过期经验或不同服务器规则混在一起。
