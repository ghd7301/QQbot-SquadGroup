from __future__ import annotations

import re


PRECISE_FACT_PATTERN = re.compile(
    r"(?P<value>\d+(?:\.\d+)?|[零〇一二两三四五六七八九十百千万两点]+)\s*"
    r"(?P<unit>公里|千米|分钟|小时|秒|米|km|m|票|人|名|倍|%|％)",
    flags=re.I,
)
CHINESE_PERCENT_PATTERN = re.compile(
    r"百分之(?P<value>[零〇一二两三四五六七八九十百千万两点]+)"
)
IPV4_PATTERN = re.compile(
    r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])"
)
VERSION_PATTERN = re.compile(
    r"(?<![\d.])(?:v|版本\s*)?(?P<value>\d+(?:\.\d+){1,2})(?![\d.])",
    flags=re.I,
)
PORT_PATTERN = re.compile(
    r"(?:端口|port)\s*[:：为是]?\s*(?P<named>\d{2,5})"
    r"|(?<![\d:])(?:[a-z0-9.-]+|\])[:：](?P<address>\d{2,5})(?!\d)",
    flags=re.I,
)


def normalize_chinese_number(value: str) -> str:
    value = str(value or "").replace("两", "二").replace("〇", "零")
    if not value or all(character.isdigit() for character in value):
        return value
    if "点" in value:
        integer, decimal = value.split("点", 1)
        decimal_digits = "".join(
            str("零一二三四五六七八九".find(character))
            for character in decimal
            if character in "零一二三四五六七八九"
        )
        return f"{normalize_chinese_number(integer or '零')}.{decimal_digits}"
    digits = {character: index for index, character in enumerate("零一二三四五六七八九")}
    units = {"十": 10, "百": 100, "千": 1000, "万": 10000}
    total = section = number = 0
    for character in value:
        if character in digits:
            number = digits[character]
            continue
        unit = units.get(character)
        if unit is None:
            return value
        if unit == 10000:
            section = (section + number) * unit
            total += section
            section = number = 0
        else:
            section += (number or 1) * unit
            number = 0
    return str(total + section + number)


def precise_fact_tokens(text: str) -> set[tuple[str, str]]:
    unit_aliases = {
        "千米": "公里",
        "km": "公里",
        "m": "米",
        "名": "人",
        "％": "%",
    }
    source = str(text or "")
    tokens = {
        (
            normalize_chinese_number(match.group("value")),
            unit_aliases.get(match.group("unit").lower(), match.group("unit").lower()),
        )
        for match in PRECISE_FACT_PATTERN.finditer(source)
    }
    tokens.update(
        (normalize_chinese_number(match.group("value")), "%")
        for match in CHINESE_PERCENT_PATTERN.finditer(source)
    )
    ip_addresses = {match.group(0) for match in IPV4_PATTERN.finditer(source)}
    tokens.update((address, "ip") for address in ip_addresses)
    tokens.update(
        (match.group("value"), "version")
        for match in VERSION_PATTERN.finditer(source)
        if match.group("value") not in ip_addresses
    )
    tokens.update(
        (match.group("named") or match.group("address"), "port")
        for match in PORT_PATTERN.finditer(source)
    )
    return tokens


def candidate_knowledge_segments(context: str) -> tuple[str, ...]:
    return tuple(
        segment.strip()
        for segment in re.split(r"\n\s*---+\s*\n", str(context or ""))
        if segment.strip()
    )


def unsupported_fallback_precise_facts(
    answer: str,
    candidate_knowledge_context: str,
) -> set[tuple[str, str]]:
    answer_facts = precise_fact_tokens(answer)
    if not answer_facts:
        return set()
    segment_facts = [
        precise_fact_tokens(segment)
        for segment in candidate_knowledge_segments(candidate_knowledge_context)
    ]
    if any(answer_facts <= facts for facts in segment_facts):
        return set()
    best_supported = max(
        segment_facts,
        key=lambda facts: len(answer_facts & facts),
        default=set(),
    )
    return answer_facts - best_supported
