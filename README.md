# Squad QQBot MVP

这是一个本地运行的《Squad / 战术小队》新兵问答 QQBot。

它是挂在 QQ 群里的“Squad 新兵营教官”：优先回答玩法、兵种、服务器、语音软件和故障排查问题，也能在白名单群里像普通群友一样低频参与闲聊。

## 当前功能

- 接收 NapCat / OneBot HTTP 上报的 QQ 群消息。
- 只处理 `ALLOWED_GROUP_IDS` 配置的白名单群。
- 支持被 `@机器人` 后优先回答。
- 白名单群普通消息也会观察，但必须像 Squad 相关问题或求助才会主动回复。
- 使用本地 Markdown 知识库检索上下文。
- 知识检索保留 Markdown 标题层级，自动从标题生成别名，对地址、时间、距离、票数和按键等精确信息提高优先级，并用查询词覆盖率阻止弱相关片段硬答。
- 知识库重载会根据片段内容哈希复用未变化的索引，并报告新增、修改、删除和复用数量。
- 将群聊消息、发送/接收时间、内容类型和生命周期持久化到独立 SQLite；按发言片段和话题分块，通过 QQ 引用链、FTS5、向量、参与者与时效混合召回。
- 长期聊天记忆只用于恢复对话承接，与 Markdown 事实知识严格分区；召回内容标记为不可信，不能改变人格、规则或第三方关系。
- 消息入库、分块和嵌入在后台线程执行；记忆数据库或嵌入服务失败时自动降级到短期上下文、引用链和可用的 FTS 结果。
- 使用检索分数和查询词覆盖率区分强命中、弱命中，避免拿弱相关资料硬答。
- 被 @ 的问题未命中知识库时，由独立的通用模型提示词谨慎兜底。
- 可在白名单群里低频参与闲聊，使用可配置的近期消息和回复关系作为上下文。
- 按群异步维护带版本、分析进度、锚点和置信度的滚动场景快照，提炼当前话题、指代关系和接话位置，不阻塞当前回复。
- 使用统一语义规划区分普通闲聊、针对机器人的调侃、人格控制尝试、第三方攻击请求、真实批评和恶意辱骂，不依赖固定触发词表。
- 群聊和机器人旧回复不能改变长期人格；轻度调侃可以等强度回一句，真实批评正常回应，第三方攻击请求直接挡回。
- 所有模型回复发送前统一检查人格、虚构经历和话题时效；生成期间的新消息会区分同话题更新、无关并行话题、已有回答和纠正，必要时最多结合最新上下文重生成一次。
- 主动闲聊在语义规划失败时不回复；被 @ 的兜底在规划失败时不读取整段原始群聊，避免上下文注入。
- 被 @ 回复和普通回复分别使用 15 秒端到端预算，模型调用不重试；超时或限流等待超过预算时放弃陈旧回复。
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
MEMBER_ID_SECRET=用于生成稳定匿名成员ID的随机字符串
ADMIN_QQ_IDS=3466734955
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
KNOWLEDGE_GAP_LOG_ENABLED=true
KNOWLEDGE_GAP_LOG=work/knowledge_gaps.jsonl
KNOWLEDGE_GAP_DEDUPE_SECONDS=3600
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
SEMANTIC_PLANNER_ENABLED=true
SEMANTIC_PLANNER_MODEL=
SEMANTIC_PLANNER_TIMEOUT_SECONDS=4
SEMANTIC_PLANNER_MIN_CONFIDENCE=0.68
CHAT_RELEVANCE_CHECK_ENABLED=true
CHAT_RELEVANCE_CHECK_MODEL=
CHAT_RELEVANCE_CHECK_TIMEOUT_SECONDS=3
FINAL_REPLY_REVIEW_MODEL=
FINAL_REPLY_REVIEW_TIMEOUT_SECONDS=4
# adaptive 可减少稳定知识问答和普通闲聊的一次模型调用；always 强制逐条终审
FINAL_REPLY_REVIEW_MODE=adaptive
MENTIONED_REPLY_TOTAL_TIMEOUT_SECONDS=15
NORMAL_REPLY_TOTAL_TIMEOUT_SECONDS=15
KNOWLEDGE_GENERATION_TIMEOUT_SECONDS=10
CHAT_GENERATION_TIMEOUT_SECONDS=7
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
CHAT_MEMORY_ENABLED=true
CHAT_MEMORY_DB=work/chat_memory.sqlite3
CHAT_MEMORY_SHADOW_MODE=false
CHAT_MEMORY_ALLOWED_GROUP_IDS=
CHAT_MEMORY_MAX_HITS=6
CHAT_MEMORY_MAX_CHARS=2400
BOT_SELF_HISTORY_MAX_TURNS=3
BOT_SELF_HISTORY_MAX_CHARS=1200
CHAT_MEMORY_PROBE_MAX_HITS=8
CHAT_MEMORY_PROBE_MAX_CHARS=1600
PLANNED_CHAT_CONTEXT_MESSAGES=10
PLANNED_CHAT_CONTEXT_MAX_CHARS=1600
CHAT_MEMORY_RETENTION_DAYS=90
CHAT_MEMORY_EMBEDDING_PROVIDER=hashed
CHAT_MEMORY_EMBEDDING_BASE_URL=
CHAT_MEMORY_EMBEDDING_API_KEY=
CHAT_MEMORY_EMBEDDING_MODEL=
CHAT_MEMORY_EMBEDDING_DIMENSIONS=384
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

`CHAT_MODEL` 控制闲聊、语义规划和默认的发送前审查，留空时复用 `LLM_MODEL`。上面的 MiMo 配置会让知识问答使用 `mimo-v2.5-pro`，闲聊、规划和审查使用 `mimo-v2.5`。具体环节仍可用各自的 `*_MODEL` 配置覆盖。

不要把 `.env` 里的真实 API Key 发到聊天、截图或公开仓库里。

`MEMBER_ID_SECRET` 只用于把同一群内的 QQ 号转换成稳定匿名成员 ID，供模型区分说话人。它不会作为用户画像使用，也不会发送真实 QQ 号；生产环境建议设置为随机字符串并保持不变。

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

普通群消息不会因为提到关键词就必然回复。知识问答、未命中兜底和闲聊使用独立提示词，统一语义规划负责判断受众、意图、候选话题和相关上下文。一条消息最多关联两个带置信度和锚点的话题，旧 `topic_id` 只作为兼容主话题；QQ 引用和机器人生成目标优先形成硬关联。闲聊只在 `CHAT_ALLOWED_GROUP_IDS` 中低频触发，并过滤公告、链接、资料整理、任务安排、bot 元讨论和群友之间的定向对话。发给模型的近期群聊会匿名化真实 QQ 号，并保留消息顺序、发送时间、内容类型、引用关系和生命周期。场景快照在后台按群合并更新；生成回复时会同时看到快照和原始上下文，并以当前消息、引用关系和原始上下文为准。模型生成后还会比较群上下文版本，决定发送、丢弃、轻量修正或最多重生成一次。

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
- 候选话题、锚点消息、场景快照版本和分析到的消息顺序。
- 近期上下文与长期记忆的候选数、实际注入数和字符预算。

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

管理命令只会对 `ADMIN_QQ_IDS` 白名单里的 QQ 进行识别。群主或群管理员身份不会自动获得权限；非白名单用户发送相同文本时，按普通群消息处理。

```text
重载知识库
健康状态
最近跳过
关闭自动回复
开启自动回复
聊天记忆状态
暂停聊天记忆
恢复聊天记忆
重建聊天记忆
清空本群聊天记忆
最近知识未命中
```

也兼容：

```text
/问 重载知识库
/问 health
/问 auto off
/问 auto on
```

关闭自动回复后，普通群消息不会主动回复；被 @ 的知识库问题仍可回答。

`清空本群聊天记忆` 是不可恢复操作，需要同一管理员在 60 秒内再次发送 `清空本群聊天记忆 确认`。这些命令仍只识别 `ADMIN_QQ_IDS` 精确白名单，目前默认仅 `3466734955`。

`CHAT_MEMORY_EMBEDDING_PROVIDER=hashed` 不需要第三方依赖，适合先运行和回放验证，但它是字符 n-gram 向量，不等同于真正语义模型。要使用兼容 OpenAI `/embeddings` 的服务，改成 `openai-compatible` 并单独配置 embedding 地址、Key 和模型；不要默认认为聊天模型套餐包含 embedding 接口。

知识 Markdown 可以在对应标题后的内容中添加可选元数据，不添加时也会正常自动索引：

```markdown
## ST 战队 TS 地址是什么
<!-- rag: aliases=TS门牌|语音地址; scope=ST战队; exact=true -->
```

字段使用分号分隔：`aliases` 是额外别名，`scope` 是适用范围，`provenance` 可覆盖自动识别的出处类型，`exact=true` 表示片段包含必须精确保留的数值、地址或按键信息。

## 已知边界

- 不建立用户偏好、情绪档案或人物关系画像。长期聊天记忆只保存消息、匿名说话人、时间、引用和话题索引，用于当前消息的按需召回。
- 闲聊回复时间写入 SQLite，因此重启服务不会重置 60 秒冷却和每小时上限。
- 主动回复策略仍需要根据真实群日志继续调优。
- 默认 `hashed` 向量更接近增强词法召回；需要更好的中文语义召回时，应接入经过验证的 embedding 服务或本地模型。
- 还没有 ST 私有知识库，涉及本服规则时应以群公告和管理员为准。
- 当前使用本机 launchd 常驻运行；源码修改后仍需先同步运行副本再重启服务。

## 后续建议

- 根据审计日志继续校准强命中阈值和闲聊接话风格。
- 先用真实群聊回放校准 `memory_needed`、召回条数和混合排序，再决定是否替换成中文语义 embedding。

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
