# Squad QQBot MVP 交接文档

更新时间：2026-07-23

## 一句话状态

这是一个运行在本机 macOS 上的《Squad / 战术小队》QQ 群 Bot。NapCat 负责登录 QQ 和 OneBot 收发，Python 服务负责知识检索、群聊上下文、模型生成、限速、审计与持久队列。目前由 launchd 常驻运行，代码和知识库修改后必须同步到隐藏运行副本再重启。

## 项目地址

源码和人工维护目录：

`/Users/yuce/Documents/Codex/2026-07-20/wo/outputs/squad-qqbot-mvp`

实际运行副本：

`/Users/yuce/.codex/qqbot-runtime/squad-qqbot-mvp`

交接文档：

`/Users/yuce/Documents/Codex/2026-07-20/wo/outputs/squad-qqbot-mvp/AGENT_HANDOFF.md`

注意：源码目录当前不是 Git 仓库。不要假设可以用 `git status`、`git diff` 或回滚提交。

## 当前运行状态

- launchd Label：`com.squad.qqbot.mvp`
- Bot 服务：`http://127.0.0.1:8088`
- 健康检查：`http://127.0.0.1:8088/health`
- NapCat OneBot API：`http://127.0.0.1:3000`
- NapCat 上报地址：`http://127.0.0.1:8088/onebot`
- NapCat WebUI 日志：`http://127.0.0.1:6099/webui/logs`
- 当前知识库：17 个 Markdown 文件，健康检查显示约 203 个切片
- 最近完整测试：53 项通过

日常使用时，Python 后端由 launchd 自动运行。用户通常只需启动 NapCat/QQ 并登录 Bot QQ。模型使用外部 API，电脑必须联网。

Mac 休眠会导致 NapCat 消息延迟到唤醒后集中送达。挂 Bot 时可运行：

```bash
caffeinate -dimsu
```

## 启动、同步和检查

进入源码目录：

```bash
cd /Users/yuce/Documents/Codex/2026-07-20/wo/outputs/squad-qqbot-mvp
```

修改源码、`.env` 或知识库后，必须同步并重启：

```bash
sh scripts/sync_runtime.sh
launchctl kickstart -k gui/$(id -u)/com.squad.qqbot.mvp
```

检查 launchd：

```bash
launchctl print gui/$(id -u)/com.squad.qqbot.mvp
```

检查服务：

```bash
curl -sS http://127.0.0.1:8088/health
```

正常结果类似：

```json
{"ok":true,"chunks":203,"queued":0,"priority_queued":0,"normal_queued":0,"chat_queued":0,"pending":0}
```

运行全部测试：

```bash
python3 -m unittest discover -s tests -v
```

仅改运行副本会在下次同步时被覆盖。人工修改必须落在源码目录。

## 关键路径

- 配置：`/Users/yuce/Documents/Codex/2026-07-20/wo/outputs/squad-qqbot-mvp/.env`
- 配置模板：`/Users/yuce/Documents/Codex/2026-07-20/wo/outputs/squad-qqbot-mvp/.env.example`
- 知识库：`/Users/yuce/Documents/Codex/2026-07-20/wo/outputs/squad-qqbot-mvp/knowledge`
- 启动入口：`run.py`
- 主服务和消息队列：`squad_bot/server.py`
- LLM 提示词和调用：`squad_bot/llm.py`
- 检索：`squad_bot/knowledge.py`
- OneBot 解析和发送：`squad_bot/onebot.py`
- 配置读取：`squad_bot/config.py`
- 同步脚本：`scripts/sync_runtime.sh`
- launchd 模板：`scripts/com.squad.qqbot.mvp.plist`
- 测试：`tests/`

运行态日志：

- stdout：`/Users/yuce/.codex/qqbot-runtime/squad-qqbot-mvp/work/launchd.out.log`
- HTTP/异常日志：`/Users/yuce/.codex/qqbot-runtime/squad-qqbot-mvp/work/launchd.err.log`
- 消息审计：`/Users/yuce/.codex/qqbot-runtime/squad-qqbot-mvp/work/message_audit.jsonl`
- 持久队列：`/Users/yuce/.codex/qqbot-runtime/squad-qqbot-mvp/work/pending_queue.sqlite3`
- 最近 OneBot 事件：`/Users/yuce/.codex/qqbot-runtime/squad-qqbot-mvp/work/last_onebot_event.json`

## 当前配置摘要

不要把 `.env` 的真实 `LLM_API_KEY` 写进聊天、文档、截图或公开仓库。

当前重要配置：

```text
BOT_QQ=3119065126
ALLOWED_GROUP_IDS=709888581,983063031,951310155
ONEBOT_API_URL=http://127.0.0.1:3000
LLM_BASE_URL=https://token-plan-cn.xiaomimimo.com/v1
LLM_MODEL=mimo-v2.5-pro
AUTO_REPLY_ENABLED=true
CHAT_REPLY_ENABLED=true
CHAT_ALLOWED_GROUP_IDS=983063031,951310155
NORMAL_MESSAGE_MAX_AGE_SECONDS=60
MENTIONED_MESSAGE_MAX_AGE_SECONDS=300
CHAT_REPLY_COOLDOWN_SECONDS=60
MAX_CHAT_REPLIES_PER_HOUR=20
CHAT_CONTEXT_SECONDS=300
CHAT_CONTEXT_MESSAGES=12
CHAT_REPLY_DEBOUNCE_SECONDS=2
```

其他阈值请直接看 `.env` 和 `squad_bot/config.py`，不要只信本文档中的历史数值。

## 系统架构

```text
QQ 群消息
  -> NapCat/QQ
  -> HTTP POST /onebot
  -> 解析文本、@、引用消息和发送者
  -> 持久化 SQLite 队列
  -> 优先问答 worker / 普通问答 worker / 闲聊 worker
  -> Markdown 知识检索或 LLM 兜底/闲聊生成
  -> OneBot send_group_msg
  -> QQ 群
```

三条回复路径：

1. 知识问答：强知识命中时严格根据知识库回答。
2. 被 @ 兜底：被 @ 但知识弱命中/未命中时，使用独立保守提示词回答；普通未 @ 消息不会随意走通用兜底。
3. 群聊闲聊：允许像普通群友一样参与日常聊天，由模型直接返回回答或 `NO_REPLY`。

## 群聊上下文实现

当前不是旧版的“只看单条消息”，已实现实时群聊上下文：

- 每群保留最近 300 秒、最多取 12 条给模型。
- 所有有效的非 Bot 群文本会在定向回复过滤前记录，因此回复其他群友的消息也能成为上下文。
- Bot 发出的内容记录为 `机器人自己：...`，避免模型把自己的上一句当成群友发言。
- 保存 QQ `message_id`、被引用消息 ID、被回复者和引用原文。
- 发给模型时明确标记 `【当前消息】`。
- 引用消息格式类似：`群友B（回复群友A“原话”）：...`。
- 候选消息进入闲聊队列后等待 2 秒；期间有新用户消息则旧候选作废。
- 模型生成期间话题继续推进，生成结果也会作废。
- 上下文在路由/生成前实时刷新，不再在消息刚入队时冻结。
- Bot 自己的消息不会被算作“话题已推进”。

关键实现位于 `squad_bot/server.py`：

- `GroupChatMessage`
- `record_group_chat_message()`
- `find_group_chat_message()`
- `recent_group_chat_context()`
- `group_chat_has_newer_user_message()`

提示词位于 `squad_bot/llm.py` 的 `CHAT_PROMPT`。

剩余限制：无引用时仍以时间顺序和同一发送者连续发言为主要线索，不是真正的语义话题聚类。高活跃群同时聊多条支线时仍可能串线，需要结合审计中的实际 `chat_context` 再判断。

## 引用回复规则

- 引用 Bot：即使没显式 @，也按直接追问处理。
- 引用其他群友且未 @ Bot：Bot 不回复，但消息仍进入群聊上下文。
- 引用其他群友同时 @ Bot：正常处理，并把引用关系交给模型。
- 本地历史找不到被引用消息时，通过 OneBot `/get_msg` 获取发送者和原文。
- OneBot 查询失败且无法确认目标时，保守跳过。
- 支持数组 `reply` segment 和字符串 `[CQ:reply,id=...]`。

## 回复与限速策略

- 被 @/管理命令使用优先 worker。
- 普通知识问答使用普通 worker。
- 闲聊使用独立 worker，不阻塞被 @ 消息。
- 普通消息过期阈值 60 秒，被 @ 消息 300 秒。
- 每分钟全局最多 8 条。
- 闲聊每群至少间隔 60 秒，每小时最多 20 条。
- 闲聊为问答预留全局回复名额。
- 同一知识主题存在主动回复冷却；被 @ 不受同主题冷却影响。
- 持久队列重启后只恢复仍在有效时间内的消息，管理命令不重放。
- 回答发送前清理 Markdown、限制长度，并把 Rally/Rally Point 统一改成“队包”。
- Bot 实现、API、提示词、回复限制、bug 等元讨论默认不主动插话。
- 单独的“哈哈”“笑死”“666”等必须有明确有效上下文，否则不接话。

## 知识库维护

知识库在源码目录的 `knowledge/`，当前 17 篇 Markdown。用户会直接人工检查和修改这些文件。

维护流程：

1. 编辑源码目录中的 Markdown。
2. 运行全部测试。
3. 执行 `sh scripts/sync_runtime.sh`。
4. 重启 launchd。
5. 用 `/health` 确认切片数量、队列和服务状态。
6. 必要时用 `shasum` 比较源码与运行副本。

群内 `重载知识库` 只会重载运行副本；如果用户改的是源码目录，仍要先执行同步脚本。

重要知识决策：

- ST TeamSpeak 地址：`GPFWD.ts5.plus`。
- “崩溃优先把 DX12 改成 DX11”属于社区经验，用户明确要求保留，但不要描述成当前版本官方通用结论。
- 当前引擎称为 Unreal Engine，不写“虚幻四”。
- 队包通俗机制：每 60 秒复活一轮；每人死亡后有强制 18 秒复活冷却。
- 放队包时 50 米内有敌军无法放置，会提示附近有敌军并进入冷却。
- 敌军进入队包 30 米范围队包消失，社区称“队包被踩了”。
- 严格区分 FOB/电台与 HAB/兵站：电台本身不是出生建筑，兵站才是全阵营出生点。

用户最近人工更新了：

- `01-新手入门.md`
- `02-出生点与工事.md`
- `05-语音软件TeamSpeak教程.md`
- `09-小队长入门.md`

最近已修正 `02-出生点与工事.md` 中的 `FO B圈` 和 `刨除地方FOB` 笔误。

`09-小队长入门.md` 当前有一处示例中文引号不成对，不影响加载，之前按用户原文保留。做文案校对前先征求用户意见，不要擅自改动社区术语和经验性表述。

## 测试覆盖

当前测试覆盖：

- 知识强/弱命中与查询覆盖率。
- FOB/HAB 不混淆。
- 队包术语规范。
- ST TS 地址检索。
- 被 @ 兜底和群聊生成。
- 裸反应消息上下文要求。
- 生日/毕业社交事件。
- 群聊上下文裁剪、Bot 自己消息、引用关系和当前消息标记。
- 新消息覆盖旧闲聊候选。
- OneBot @、reply segment、引用目标查询和结构化发送。
- 普通/被 @ 消息不同过期时间。
- SQLite 待处理队列恢复。
- 管理命令不重放。

基线命令：

```bash
python3 -m unittest discover -s tests -v
```

交接时最近结果：53 项全部通过。

## 审计与排障

优先查 `work/message_audit.jsonl`，每行记录：

- 群号、用户、原问题、是否 @。
- `decision` 和 `reason`。
- 回复模式：knowledge/fallback/chat。
- 检索分数、覆盖率、来源。
- 模型耗时。
- 实际传给闲聊模型的 `chat_context`。
- 最终闲聊回答。
- 引用消息 ID 和目标用户。

常见原因：

- `chat candidate superseded`：2 秒 debounce 内出现新消息。
- `chat superseded during generation`：生成期间话题继续推进。
- `chat became stale during generation`：模型太慢，结果超过消息有效期。
- `chat cooldown` / `chat hourly limit`：闲聊额度限制。
- `reply directed at another member`：引用的是其他群友，Bot 不插话。
- `reply target unknown`：OneBot 无法解析引用目标。
- `weak knowledge context for factual question`：知识命中不足，不允许普通消息硬答。

诊断答非所问时，先找到对应审计记录，检查“当前消息 + chat_context + answer”，不要先调大上下文窗口。更大的纯时间窗口会增加串话题风险。

## 已确认的历史问题

### 消息延迟

曾出现 18:11 后的消息到 18:28/18:42 才集中到达。系统电源日志确认是 Mac 休眠/暗唤醒，不是模型推理耗时。不要通过放宽过期消息阈值解决，应保持 Mac 唤醒。

### “当蛆”后自己接错话

不是同一消息重复处理。原因是模型当时看不到 Bot 自己上一句，只看到后续“60s 回复冷却”等消息，错误连接到“怎么当蛆”。现已记录 `机器人自己` 并加入话题推进保护。

### TS 地址答错

原知识检索没有覆盖“ts网址是什么”。已从用户 PDF 提取并新增专门 FAQ 和别名，地址为 `GPFWD.ts5.plus`。

### NapCat 400

NapCat HTTP 上报可能使用 chunked body，早期服务读取不兼容。`Handler._read_request_body()` 已支持 chunked。

## 已知风险与后续方向

1. 群聊上下文仍不是完整语义线程系统；无引用的并行话题可能串线。
2. 模型请求偶尔需要 10 到 30 秒，活跃群里回答经常因话题推进而被丢弃。这是当前保守设计，不应直接取消 supersede 检查。
3. 群聊历史在内存中，服务重启后不会恢复；SQLite 只持久化待处理消息和部分限速状态。
4. 审计和 launchd 日志没有轮转，长期运行需要加轮转策略。
5. 关键词/BM25 风格检索没有 embedding，语义改写问法仍有边界。
6. NapCat 曾出现 QQ arm64 版本架构警告，虽然 OneBot HTTP 收发可用，升级 QQ/NapCat 后需要回归测试。
7. 源码目录不是 Git 仓库，重要修改缺少版本历史，建议后续初始化私有仓库并确保 `.env`、日志、SQLite 不提交。
8. README 部分描述仍可能落后于关系型群聊上下文，以代码和本文档为准。

## 下一位 Agent 的工作原则

- 先读真实审计记录，再判断是知识库、路由、上下文还是模型生成问题。
- 用户提供的社区经验需要保留；如果缺少官方依据，应标注为社区经验，不要直接删除。
- 不要把游戏版本相关数值擅自改成绝对事实。
- 用户人工编辑的知识库尽量原样保留；只修明确笔误或结构错误，并说明修改。
- 修改后必须跑测试、同步运行副本、重启并健康检查。
- 不泄露 `.env`、API Key、访问令牌。
- 不清空用户日志、队列或知识库，不使用破坏性命令。
- 本机 `curl 127.0.0.1`、launchctl 或写入运行副本在受限环境中可能需要权限提升。

## 快速接手清单

```bash
cd /Users/yuce/Documents/Codex/2026-07-20/wo/outputs/squad-qqbot-mvp
python3 -m unittest discover -s tests -v
launchctl print gui/$(id -u)/com.squad.qqbot.mvp
curl -sS http://127.0.0.1:8088/health
tail -n 50 /Users/yuce/.codex/qqbot-runtime/squad-qqbot-mvp/work/message_audit.jsonl
```

如果只想日常启动：确认 Python launchd 服务健康，然后启动 NapCat/QQ、登录 Bot 账号，并保持 Mac 不休眠。
