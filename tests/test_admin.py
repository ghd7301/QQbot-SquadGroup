import unittest
from types import SimpleNamespace

from squad_bot import admin, server


class AdminTests(unittest.TestCase):
    def test_command_normalization_uses_live_aliases(self) -> None:
        deps = SimpleNamespace(
            normalize_command_text=lambda message: admin.normalize_command_text(
                SimpleNamespace(), message
            ),
            COMMAND_ALIASES={"memory status": "memory_status"},
        )

        command = admin.get_admin_command(deps, "  MEMORY   STATUS ")

        self.assertEqual(command, "memory_status")

    def test_restored_command_uses_injected_admin_checks(self) -> None:
        deps = SimpleNamespace(
            is_admin_user=lambda user_id, _role: str(user_id) == "100",
            get_admin_command=lambda question: "health" if question == "健康状态" else "",
        )

        allowed = admin.is_restored_admin_command(
            deps,
            {
                "_restored": True,
                "user_id": "100",
                "sender_role": "member",
                "question": "健康状态",
            },
        )

        self.assertTrue(allowed)

    def test_server_auto_reply_command_writes_live_state(self) -> None:
        original = server.auto_reply_enabled
        try:
            self.assertEqual(
                server.answer_admin_command("auto_reply_off"),
                "自动回复已关闭。被 @ 时仍可回答。",
            )
            self.assertFalse(server.auto_reply_enabled)

            self.assertEqual(
                server.answer_admin_command("auto_reply_on"),
                "自动回复已开启。",
            )
            self.assertTrue(server.auto_reply_enabled)
        finally:
            server.auto_reply_enabled = original


if __name__ == "__main__":
    unittest.main()
