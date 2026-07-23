# launchd 常驻运行

这个项目可以用 macOS 自带的 launchd 常驻运行。

macOS 可能限制 launchd 读取 `Documents` 目录，所以常驻服务建议从运行副本启动：

```text
/Users/yuce/.codex/qqbot-runtime/squad-qqbot-mvp
```

## 安装

```bash
mkdir -p ~/Library/LaunchAgents
sh scripts/sync_runtime.sh
cp scripts/com.squad.qqbot.mvp.plist ~/Library/LaunchAgents/com.squad.qqbot.mvp.plist
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.squad.qqbot.mvp.plist
launchctl enable gui/$(id -u)/com.squad.qqbot.mvp
launchctl kickstart -k gui/$(id -u)/com.squad.qqbot.mvp
```

## 更新运行副本

修改源码、配置或知识库后，从项目目录执行：

```bash
sh scripts/sync_runtime.sh
launchctl kickstart -k gui/$(id -u)/com.squad.qqbot.mvp
```

## 查看状态

```bash
launchctl print gui/$(id -u)/com.squad.qqbot.mvp
curl http://127.0.0.1:8088/health
```

## 日志

```text
~/.codex/qqbot-runtime/squad-qqbot-mvp/work/launchd.out.log
~/.codex/qqbot-runtime/squad-qqbot-mvp/work/launchd.err.log
~/.codex/qqbot-runtime/squad-qqbot-mvp/work/message_audit.jsonl
```

## 卸载

```bash
launchctl bootout gui/$(id -u)/com.squad.qqbot.mvp
rm ~/Library/LaunchAgents/com.squad.qqbot.mvp.plist
```
