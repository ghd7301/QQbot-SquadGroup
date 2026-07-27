import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


TOKEN_RE = re.compile(r"[\u4e00-\u9fff]+|[a-zA-Z0-9_+#.-]+")
RAG_META_RE = re.compile(r"<!--\s*rag:\s*(.*?)-->", re.I | re.S)
EXACT_VALUE_RE = re.compile(
    r"(?i)(?:(?<![/\w])[a-z0-9.-]+\.(?:com|cn|net|org|plus)\b|\d+(?:\.\d+)?\s*(?:米|m|秒|s|分钟|min|票|发|人|倍|g|gb|mb)|\b[VGBC]\s*键\b)"
)
EXACT_QUERY_CUES = (
    "多少", "多久", "几秒", "几分钟", "多远", "距离", "范围", "冷却",
    "地址", "网址", "服务器", "按键", "哪个键", "多少票", "多少米",
)

QUESTION_STOP_TOKENS = {
    "是什",
    "什么",
    "是什么",
    "是啥",
    "啥啊",
    "是啥啊",
    "啥意",
    "么意",
    "意思",
    "思是",
    "啥意思",
    "什么意思",
    "有什",
    "么区",
    "区别",
    "有什么",
    "什么区",
    "么区别",
    "怎么",
    "么办",
    "怎么办",
    "咋办",
    "为什",
    "什么",
    "为什么",
    "咋玩",
    "么玩",
    "怎么玩",
    "什么意",
    "么意思",
    "兵是",
    "兵是什",
    "阴兵是",
    "种是",
    "兵种是",
    "种是什",
    "键有",
    "键有什",
}

QUESTION_NOISE_CHARS = "是什啥怎么意了吗啊呢呀有"

QUERY_COVERAGE_NOISE = (
    "上一轮问题",
    "当前追问",
    "为什么",
    "怎么办",
    "怎么处理",
    "怎么说",
    "怎么用",
    "怎么玩",
    "怎么样",
    "怎么",
    "咋玩",
    "咋办",
    "如何",
    "是什么",
    "是不是",
    "能不能",
    "可不可以",
    "可以吗",
    "什么意思",
    "有啥区别",
    "有什么区别",
    "区别",
    "请问",
    "问一下",
    "请教",
    "求助",
    "这个",
    "那个",
    "那",
    "所以",
    "还有",
    "以及",
    "然后",
    "分别",
    "一下",
    "要",
    "和",
    "与",
    "在哪",
    "哪里",
    "多少",
    "多久",
    "为啥",
    "什么",
    "是",
    "的",
    "啊",
    "呢",
    "吗",
)

COLLOQUIAL_QUERY_REWRITES = (
    ("咋样", "怎么样"),
    ("咋个", "怎么"),
    ("咋", "怎么"),
)

QUERY_ALIASES = (
    (
        (
            "ts去哪下载",
            "ts3去哪下载",
            "teamspeak去哪下载",
            "ts安装包",
            "ts3安装包",
            "teamspeak安装包",
            "ts汉化包",
            "ts3汉化包",
            "teamspeak汉化包",
        ),
        "TeamSpeak 3 TS3 本体 安装包 汉化包 QQ 群文件 下载",
    ),
    (
        (
            "ts网址",
            "ts地址",
            "st战队ts",
            "st战队语音",
            "语音地址",
            "teamspeak地址",
            "teamspeak服务器",
        ),
        "ST 战队 TS TeamSpeak 3 语音服务器 地址 网址 GPFWD.ts5.plus",
    ),
    (
        ("rally", "rally point"),
        "队包 小队包 临时出生点 小队临时出生点",
    ),
    (
        ("卡fob", "卡 fob", "fob圈", "fob 圈"),
        "卡 FOB 圈 互相卡圈 排斥圈 排斥半径 灰圈 白圈 不能再放另一个己方 Radio 不能放第二个电台",
    ),
    (
        ("fob和hab", "fob与hab", "fob是不是兵站", "fob是兵站吗"),
        "FOB 电台 Radio HAB 兵站 区别 建设根基 出生建筑 不是同一个",
    ),
    (
        ("电台有什么用", "fob有什么用", "电台是啥", "电台是什么"),
        "FOB 电台 Radio 建筑圈 建设点 工事根基 不能出生",
    ),
    (
        (
            "兵站为什么不能复活",
            "hab为什么不能复活",
            "围攻中重生",
            "兵站被压",
            "hab被压",
        ),
        "HAB 兵站 不能复活 围攻 压制 至少2名敌人 敌人越多范围越大",
    ),
    (
        ("乱报坦克", "报坦克", "不要乱报坦克"),
        "报载具 载具报点 轮式 履带 炮塔 长炮管 短炮管 轻甲 真正敌坦",
    ),
    (
        ("bvg", "b v g", "b键v键g键", "b键 v键 g键"),
        "B键 V键 G键 语音颜色 小队语音 本地语音 指挥频道",
    ),
    (
        ("压家圈", "压家", "maincamp", "main camping"),
        "压家 主基地保护 压家圈 服务器规则 主基地出入口 载具保护 禁止压家",
    ),
    (
        ("单载", "solo armor", "solo vehicle"),
        "单载 一个人驾驶载具 载具队 人数要求 服务器规则 车辆归属权",
    ),
    (
        ("特装", "大特", "重筒", "特殊装备"),
        "特装限制 重反坦 狙击手 工兵 通用机枪 精确射手 兵种优先级",
    ),
    (
        ("拉双白", "双白", "点不动"),
        "双白 防守点 进攻点 白旗 点内人数 人数差 至少多出三人 回防",
    ),
    (
        ("拉点速度", "几倍速", "一倍速", "五倍速"),
        "拉点速度 有效人数 箭头 倍速 点内人数 拉白旗",
    ),
    (
        ("体力条", "耐力条", "稳枪"),
        "体力条 耐力 举枪稳定 瞄准恢复 跑一段走一段 接敌",
    ),
    (
        ("断履", "打履带", "履带"),
        "反坦 断履 打履带 履带式载具 炮塔 发动机 限制机动",
    ),
    (
        ("侦察车", "侦查车", "侦察车队"),
        "侦察车 信息收集 截补给线 绕后 后勤线 拉点队 载具路线",
    ),
    (
        ("步战协同", "载具协同"),
        "步战协同 步兵保护载具 载具压制 步兵看反坦 信息共享",
    ),
    (
        ("步战", "步战车", "步兵战车", "ifv", "IFV"),
        "步兵战车 IFV 载具 步战协同 遇到坦克 撤退 反打 履带 炮塔",
    ),
    (
        ("特射", "精确射手怎么玩", "精确射手干嘛的"),
        "精确射手 Marksman 侦察 报点 压制 跟队 远距离 不是狙击手",
    ),
    (
        ("重桶", "重筒怎么打", "重反怎么用", "hat怎么玩"),
        "重反坦 HAT 重型反坦克 关键资源 距离 弹药 补给 测距",
    ),
    (
        ("补给卡", "补给车", "logi", "后勤车怎么开"),
        "后勤车 补给卡 Logi 运输 建设点 弹药点 卸货 路线 FOB",
    ),
    (
        ("压制范围", "兵站压制", "围攻范围", "围攻距离"),
        "HAB 兵站 围攻 压制 至少2名敌人 敌人越多范围越大 不能复活",
    ),
)

PHRASE_BOOSTS = (
    (
        (
            "ts去哪下载",
            "ts3去哪下载",
            "teamspeak去哪下载",
            "ts安装包",
            "ts3安装包",
            "teamspeak安装包",
            "ts汉化包",
            "ts3汉化包",
            "teamspeak汉化包",
        ),
        ("teamspeak 3安装包和汉化包在哪里下载",),
        1.2,
    ),
    (
        ("ts网址", "ts地址", "st战队ts", "语音地址", "teamspeak服务器"),
        ("st战队ts地址是什么", "ts3怎么连接服务器"),
        1.2,
    ),
    (
        ("fob和hab", "fob与hab", "fob是不是兵站", "fob是兵站吗"),
        ("fob是什么", "hab是什么", "fob、电台、兵站、hab是什么关系"),
        1.0,
    ),
    (
        ("电台有什么用", "fob有什么用", "电台是什么"),
        ("fob是什么", "fob、电台、兵站、hab是什么关系"),
        0.9,
    ),
    (
        ("兵站为什么不能复活", "hab为什么不能复活", "围攻中重生", "兵站被压", "hab被压"),
        ("hab兵站为什么不能复活", "兵站被压怎么救"),
        1.0,
    ),
    (
        ("卡fob", "fob圈", "卡fob圈"),
        ("卡fob圈", "卡fob", "互相卡圈", "排斥圈", "排斥半径"),
        1.0,
    ),
    (
        ("乱报坦克", "报坦克", "不要乱报坦克"),
        ("不要看见什么都喊坦克", "为什么不要乱报坦克", "新手该怎么报载具"),
        0.8,
    ),
    (
        ("压家圈", "压家"),
        ("压家和压家圈是什么", "压家圈"),
        0.9,
    ),
    (
        ("单载",),
        ("载具队人数上限和归属权怎么看", "单载"),
        0.7,
    ),
    (
        ("拉双白", "双白"),
        ("双白是什么意思", "双方交汇后为什么点不动"),
        0.9,
    ),
    (
        ("侦察车", "侦查车"),
        ("侦察车应该干什么", "侦察车"),
        0.7,
    ),
    (
        ("步战", "步战车", "步兵战车"),
        ("步战协同", "载具协同", "步兵保护载具"),
        0.8,
    ),
    (
        ("特射",),
        ("精确射手", "Marksman 侦察 报点"),
        0.8,
    ),
    (
        ("重桶", "重筒", "重反", "hat"),
        ("重反坦", "HAT 重型反坦克"),
        0.8,
    ),
    (
        ("补给卡", "补给车", "logi", "后勤车"),
        ("后勤车", "补给卡", "Logi 运输"),
        0.8,
    ),
    (
        ("压制范围", "兵站压制", "围攻范围"),
        ("HAB 兵站 围攻 压制", "兵站为什么不能复活"),
        0.9,
    ),
)


def is_question_noise_token(token: str) -> bool:
    if token in QUESTION_STOP_TOKENS:
        return True
    if re.fullmatch(r"[\u4e00-\u9fff]+", token):
        return any(char in token for char in QUESTION_NOISE_CHARS)
    return False


def expand_query(query: str) -> str:
    lowered = query.lower().replace(" ", "")
    additions: list[str] = []
    for triggers, alias_text in QUERY_ALIASES:
        if any(trigger.replace(" ", "") in lowered for trigger in triggers):
            additions.append(alias_text)
    if not additions:
        return query
    return query + "\n" + "\n".join(additions)


def compact_text(text: str) -> str:
    return re.sub(r"\s+", "", text.lower())


def contains_exact_value(text: str) -> bool:
    without_source_urls = re.sub(r"https?://\S+", "", str(text or ""), flags=re.I)
    return bool(EXACT_VALUE_RE.search(without_source_urls))


@dataclass
class Chunk:
    source: str
    title: str
    text: str
    section_path: str = ""
    aliases: tuple[str, ...] = ()
    provenance: str = "maintained"
    scope: str = "general"
    exact_fact: bool = False
    content_hash: str = ""


@dataclass
class ContextResult:
    context: str
    sources: list[str]
    top_score: float
    query_coverage: float
    matched_query_tokens: tuple[str, ...] = ()
    missing_query_tokens: tuple[str, ...] = ()
    exact_match: bool = False


@dataclass(frozen=True)
class ReloadStats:
    total: int = 0
    added: int = 0
    changed: int = 0
    removed: int = 0
    reused: int = 0


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for raw in TOKEN_RE.findall(text):
        token = raw.lower()
        if re.fullmatch(r"[\u4e00-\u9fff]+", token):
            if len(token) == 1:
                continue
            tokens.extend(token[index : index + 2] for index in range(len(token) - 1))
            if len(token) >= 3:
                tokens.extend(token[index : index + 3] for index in range(len(token) - 2))
        else:
            tokens.append(token)
    return tokens


def coverage_tokens(text: str) -> set[str]:
    cleaned = text.lower()
    for colloquial, standard in COLLOQUIAL_QUERY_REWRITES:
        cleaned = cleaned.replace(colloquial, standard)
    for phrase in QUERY_COVERAGE_NOISE:
        cleaned = cleaned.replace(phrase, " ")

    tokens: set[str] = set()
    for raw in TOKEN_RE.findall(cleaned):
        token = raw.lower()
        if re.fullmatch(r"[\u4e00-\u9fff]+", token):
            if len(token) < 2:
                continue
            tokens.update(token[index : index + 2] for index in range(len(token) - 1))
        elif token:
            tokens.add(token)
    return tokens


def _parse_rag_metadata(text: str) -> dict[str, str]:
    match = RAG_META_RE.search(text)
    if not match:
        return {}
    metadata: dict[str, str] = {}
    for item in match.group(1).split(";"):
        key, separator, value = item.partition("=")
        if separator and key.strip():
            metadata[key.strip().lower()] = value.strip()
    return metadata


def _title_aliases(title: str, section_path: str, explicit: str = "") -> tuple[str, ...]:
    candidates: list[str] = []
    if explicit:
        candidates.extend(re.split(r"[|,，、]", explicit))
    candidates.extend(re.split(r"[/／、｜|（）()：:]", title))
    candidates.extend(part.strip() for part in section_path.split(" > "))
    compact = compact_text(title)
    for suffix in ("是什么", "是什么意思", "怎么做", "怎么玩", "怎么用", "怎么办"):
        if compact.endswith(suffix) and len(compact) > len(suffix):
            candidates.append(compact[: -len(suffix)])
    aliases: list[str] = []
    for candidate in candidates:
        value = candidate.strip()
        if len(value) >= 2 and value != title and value not in aliases:
            aliases.append(value)
    return tuple(aliases)


def _infer_provenance(raw: str) -> str:
    lowered = raw.lower()
    has_official = "squad wiki" in lowered or "官方" in raw
    has_local = "本地 pdf" in lowered or "服务器规则" in raw or "战队" in raw
    has_community = "社区经验" in raw or "小黑盒" in raw or "b 站" in lowered
    kinds = [name for name, enabled in (("official", has_official), ("local", has_local), ("community", has_community)) if enabled]
    return "+".join(kinds) if kinds else "maintained"


def split_markdown(path: Path) -> list[Chunk]:
    raw = path.read_text(encoding="utf-8")
    parts: list[Chunk] = []
    current_title = path.stem
    current_lines: list[str] = []
    current_level = 1
    heading_stack: list[tuple[int, str]] = []
    provenance = _infer_provenance(raw)

    def flush() -> None:
        body = "\n".join(current_lines).strip()
        if body:
            metadata = _parse_rag_metadata(body)
            clean_body = RAG_META_RE.sub("", body).strip()
            section_path = " > ".join(title for _level, title in heading_stack)
            aliases = _title_aliases(current_title, section_path, metadata.get("aliases", ""))
            digest_source = "\n".join((path.name, section_path, clean_body, "|".join(aliases)))
            parts.append(Chunk(
                source=path.name,
                title=current_title,
                text=clean_body,
                section_path=section_path or current_title,
                aliases=aliases,
                provenance=metadata.get("provenance", provenance),
                scope=metadata.get("scope", "general"),
                exact_fact=metadata.get("exact", "").lower() in {"1", "true", "yes"}
                or contains_exact_value(clean_body),
                content_hash=hashlib.sha256(digest_source.encode("utf-8")).hexdigest(),
            ))

    for line in raw.splitlines():
        heading = re.match(r"^(#{1,6})\s+(.*)$", line)
        if heading:
            flush()
            current_level = len(heading.group(1))
            current_title = heading.group(2).strip() or path.stem
            while heading_stack and heading_stack[-1][0] >= current_level:
                heading_stack.pop()
            heading_stack.append((current_level, current_title))
            current_lines = [line]
        else:
            current_lines.append(line)
    flush()
    return parts


class KnowledgeBase:
    def __init__(self, root: str):
        self.root = Path(root)
        self.chunks: list[Chunk] = []
        self.doc_freq: dict[str, int] = {}
        self.chunk_tokens: list[list[str]] = []
        self.chunk_title_tokens: list[set[str]] = []
        self.chunk_alias_tokens: list[set[str]] = []
        self.last_reload_stats = ReloadStats()
        self.reload()

    def reload(self) -> int:
        previous = {
            (chunk.source, chunk.section_path): (chunk.content_hash, tokens)
            for chunk, tokens in zip(self.chunks, self.chunk_tokens)
        }
        next_chunks: list[Chunk] = []
        next_tokens: list[list[str]] = []
        added = changed = reused = 0

        for path in sorted(self.root.glob("**/*.md")):
            for chunk in split_markdown(path):
                key = (chunk.source, chunk.section_path)
                cached = previous.get(key)
                if cached and cached[0] == chunk.content_hash:
                    tokens = cached[1]
                    reused += 1
                else:
                    tokens = tokenize(f"{chunk.section_path}\n{' '.join(chunk.aliases)}\n{chunk.text}")
                    if cached:
                        changed += 1
                    else:
                        added += 1
                next_chunks.append(chunk)
                next_tokens.append(tokens)
        next_keys = {(chunk.source, chunk.section_path) for chunk in next_chunks}
        removed = len(set(previous) - next_keys)
        self.chunks = next_chunks
        self.chunk_tokens = next_tokens
        self.chunk_title_tokens = [set(tokenize(chunk.section_path)) for chunk in self.chunks]
        self.chunk_alias_tokens = [set(tokenize(" ".join(chunk.aliases))) for chunk in self.chunks]
        self.doc_freq.clear()
        for tokens in self.chunk_tokens:
            for token in set(tokens):
                self.doc_freq[token] = self.doc_freq.get(token, 0) + 1
        self.last_reload_stats = ReloadStats(len(self.chunks), added, changed, removed, reused)
        return len(self.chunks)

    def search(self, query: str, limit: int = 5, min_score: float = 0.08) -> list[tuple[Chunk, float]]:
        query_tokens = tokenize(expand_query(query))
        if not query_tokens:
            return []
        compact_query = compact_text(query)
        exact_query = any(cue in compact_query for cue in EXACT_QUERY_CUES) or bool(EXACT_VALUE_RE.search(query))

        total_docs = max(len(self.chunks), 1)
        query_set = set(query_tokens)
        meaningful_query_set = {
            token for token in query_set if not is_question_noise_token(token)
        }
        if meaningful_query_set:
            query_set = meaningful_query_set
        else:
            return []

        scored: list[tuple[Chunk, float]] = []

        for chunk, tokens, title_tokens, alias_tokens in zip(
            self.chunks, self.chunk_tokens, self.chunk_title_tokens, self.chunk_alias_tokens
        ):
            if not tokens:
                continue
            token_count = len(tokens)
            counts: dict[str, int] = {}
            for token in tokens:
                if token in query_set:
                    counts[token] = counts.get(token, 0) + 1
            score = 0.0
            for token, count in counts.items():
                tf = count / token_count
                idf = math.log((total_docs + 1) / (self.doc_freq.get(token, 0) + 1)) + 1
                score += tf * idf
                if re.fullmatch(r"[a-z0-9_+#.-]+", token):
                    score += 0.08

            title = chunk.title.lower()
            text = chunk.text.lower()
            compact_title = compact_text(chunk.title)
            compact_body = compact_text(chunk.text)
            compact_aliases = tuple(compact_text(alias) for alias in chunk.aliases)
            title_overlap = len(query_set.intersection(title_tokens))
            alias_overlap = len(query_set.intersection(alias_tokens))
            score += title_overlap * 0.18 + alias_overlap * 0.22
            if compact_query and compact_query in compact_title:
                score += 0.65
            if any(alias and (alias in compact_query or compact_query in alias) for alias in compact_aliases):
                score += 0.45
            if any(cue in compact_query for cue in ("是什么", "是啥", "什么意思")):
                if any(cue in compact_title for cue in ("是什么", "是啥", "什么意思")):
                    score += 0.28
            for token in query_set:
                if token in title:
                    if re.fullmatch(r"[a-z0-9_+#.-]+", token):
                        if len(token) >= 2:
                            score += 0.35
                    else:
                        score += 0.24
                elif re.fullmatch(r"[a-z0-9_+#.-]+", token) and len(token) >= 2 and token in text:
                    score += 0.04

            for triggers, targets, boost in PHRASE_BOOSTS:
                if any(trigger in compact_query for trigger in triggers):
                    if any(target in compact_title for target in targets):
                        score += boost
                    elif any(target in compact_body for target in targets):
                        score += boost * 0.5

            if exact_query and chunk.exact_fact and counts:
                score += 0.45
                query_exact_values = set(EXACT_VALUE_RE.findall(query))
                if query_exact_values and query_exact_values.intersection(EXACT_VALUE_RE.findall(chunk.text)):
                    score += 0.5

            if score >= min_score:
                scored.append((chunk, score))

        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:limit]

    def build_context(self, query: str, max_chars: int) -> tuple[str, list[str]]:
        result = self.build_context_with_metrics(query, max_chars)
        return result.context, result.sources

    def build_context_with_metrics(self, query: str, max_chars: int) -> ContextResult:
        matches = self.search(query)
        matches = self._select_diverse_matches(query, matches)
        compact_query = compact_text(query)
        exact_query = any(cue in compact_query for cue in EXACT_QUERY_CUES) or bool(EXACT_VALUE_RE.search(query))
        context_parts: list[str] = []
        sources: list[str] = []
        used = 0

        for chunk, _score in matches:
            attributes = [f"出处类型：{chunk.provenance}"]
            if chunk.scope != "general":
                attributes.append(f"适用范围：{chunk.scope}")
            if chunk.exact_fact:
                attributes.append("包含需精确保持的数值、地址或按键信息")
            block = (
                f"来源：{chunk.source} / {chunk.section_path}\n"
                f"资料属性：{'；'.join(attributes)}\n{chunk.text.strip()}\n"
            )
            if used + len(block) > max_chars:
                block = block[: max(0, max_chars - used)]
            if not block.strip():
                continue
            context_parts.append(block)
            sources.append(f"{chunk.source} / {chunk.title}")
            used += len(block)
            if used >= max_chars:
                break

        top_score = matches[0][1] if matches else 0.0
        query_tokens = coverage_tokens(query)
        query_coverage = 0.0
        matched_tokens: set[str] = set()
        if matches and query_tokens:
            context_tokens: set[str] = set()
            matched_chunks = {id(chunk) for chunk, _score in matches}
            for chunk, tokens in zip(self.chunks, self.chunk_tokens):
                if id(chunk) in matched_chunks:
                    context_tokens.update(tokens)
            matched_tokens = query_tokens.intersection(context_tokens)
            query_coverage = len(matched_tokens) / len(query_tokens)

        return ContextResult(
            context="\n---\n".join(context_parts),
            sources=sources,
            top_score=top_score,
            query_coverage=query_coverage,
            matched_query_tokens=tuple(sorted(matched_tokens)),
            missing_query_tokens=tuple(sorted(query_tokens - matched_tokens)),
            exact_match=bool(exact_query and matches and matches[0][0].exact_fact),
        )

    def _select_diverse_matches(
        self,
        query: str,
        matches: Sequence[tuple[Chunk, float]],
    ) -> list[tuple[Chunk, float]]:
        if len(matches) <= 1:
            return list(matches)
        query_tokens = coverage_tokens(query)
        selected: list[tuple[Chunk, float]] = []
        covered: set[str] = set()
        remaining = list(matches)
        while remaining and len(selected) < 5:
            best_index = 0
            best_value = float("-inf")
            for index, (chunk, score) in enumerate(remaining):
                chunk_terms = set(tokenize(f"{chunk.section_path}\n{' '.join(chunk.aliases)}\n{chunk.text}"))
                new_coverage = len((query_tokens - covered).intersection(chunk_terms)) / max(1, len(query_tokens))
                source_bonus = 0.04 if all(existing.source != chunk.source for existing, _ in selected) else 0.0
                value = score + 0.35 * new_coverage + source_bonus
                if value > best_value:
                    best_index, best_value = index, value
            chunk, score = remaining.pop(best_index)
            selected.append((chunk, score))
            covered.update(query_tokens.intersection(tokenize(f"{chunk.section_path}\n{chunk.text}")))
        return selected
