from __future__ import annotations

import time


def normalize_command_text(deps, message: str) -> str:
    return " ".join(message.strip().lower().split())


def get_admin_command(deps, message: str) -> str:
    normalized = deps.normalize_command_text(message)
    return deps.COMMAND_ALIASES.get(normalized, "")


def is_restored_admin_command(deps, item: dict) -> bool:
    return bool(
        item.get("_restored")
        and deps.is_admin_user(item.get("user_id"), item.get("sender_role", ""))
        and deps.get_admin_command(str(item.get("question", "")))
    )


def is_admin_user(deps, user_id, sender_role: str = "") -> bool:
    admin_ids = tuple(getattr(deps.settings, "admin_qq_ids", ()))
    return bool(admin_ids) and str(user_id) in admin_ids


def answer_admin_command(
    deps, command: str, *, group_id: int = 0, user_id: str = ""
) -> str:
    if command == "reload":
        with deps.kb_lock:
            count = deps.kb.reload()
            stats = deps.kb.last_reload_stats
        return f"知识库已重载，共 {count} 个片段；新增 {stats.added}，修改 {stats.changed}，删除 {stats.removed}，复用 {stats.reused}。"
    if command == "health":
        auto_status = "开" if deps.auto_reply_enabled else "关"
        chat_status = (
            "开"
            if deps.settings.chat_reply_enabled and deps.auto_reply_enabled
            else "关"
        )
        queued = (
            deps.message_queue.qsize()
            + deps.normal_message_queue.qsize()
            + deps.chat_queue.qsize()
        )
        (scene_count, _) = deps.chat_scene_state.counts()
        return f"服务正常。知识片段 {len(deps.kb.chunks)} 个，队列 {queued} 条，自动回复{auto_status}，闲聊{chat_status}，场景快照 {scene_count} 个，每分钟最多回复 {deps.settings.max_replies_per_minute} 条。"
    if command == "recent_skips":
        entries = deps.recent_audit_entries()
        if not entries:
            return "最近还没有记录到跳过消息。"
        parts: list[str] = []
        for entry in entries:
            question = str(entry.get("question") or "").strip()
            if len(question) > 28:
                question = question[:28] + "..."
            reason = entry.get("reason") or "unknown"
            parts.append(
                f"{entry.get('group_id') or '未知群'}：{reason}：{question or '空消息'}"
            )
        return "最近跳过消息：\n" + "\n".join(parts)
    if command == "auto_reply_on":
        deps.auto_reply_enabled = True
        return "自动回复已开启。"
    if command == "auto_reply_off":
        deps.auto_reply_enabled = False
        return "自动回复已关闭。被 @ 时仍可回答。"
    if command == "memory_status":
        if not deps.chat_memory_manager:
            return "聊天记忆没有启用。"
        status = deps.chat_memory_manager.status()
        state = "暂停" if status["paused"] else "运行"
        return f"聊天记忆{state}，消息 {status['messages']} 条，片段 {status['chunks']} 个，话题关系 {status.get('topic_relations', 0)} 条，待索引 {status['queued']} 条，向量方式 {status['provider']}。"
    if command == "memory_pause":
        if not deps.chat_memory_manager:
            return "聊天记忆没有启用。"
        deps.chat_memory_manager.paused.set()
        return "聊天记忆索引已暂停，现有短期上下文仍然可用。"
    if command == "memory_resume":
        if not deps.chat_memory_manager:
            return "聊天记忆没有启用。"
        deps.chat_memory_manager.paused.clear()
        return "聊天记忆索引已恢复。"
    if command == "memory_rebuild":
        if not deps.chat_memory_manager:
            return "聊天记忆没有启用。"
        deps.chat_memory_manager.enqueue_rebuild(group_id)
        return "已安排重建本群聊天记忆索引。"
    if command == "memory_clear_request":
        deps.memory_clear_confirmations[group_id, str(user_id)] = time.time() + 60
        return "这是不可恢复操作。确认要清空本群聊天记忆，请发送：清空本群聊天记忆 确认"
    if command == "memory_clear_confirm":
        if not deps.chat_memory_manager:
            return "聊天记忆没有启用。"
        key = (group_id, str(user_id))
        expires_at = deps.memory_clear_confirmations.pop(key, 0)
        if time.time() > expires_at:
            return "确认已失效，请先发送：清空本群聊天记忆"
        deps.chat_memory_manager.store.clear_group(group_id)
        return "本群聊天记忆已清空。"
    if command == "knowledge_gaps":
        entries = deps.recent_knowledge_gap_entries()
        if not entries:
            return "最近没有记录到知识检索缺口。"
        lines = []
        for entry in entries:
            query = str(entry.get("query") or "")[:36]
            missing = "、".join(entry.get("missing_tokens") or ())[:50]
            lines.append(f"{query}（缺：{missing or '无明确词'}）")
        return "最近知识未命中：\n" + "\n".join(lines)
    return "未知维护命令。"
