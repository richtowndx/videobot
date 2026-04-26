# VideoBot

一个 Telegram 机器人，接收 Bilibili / YouTube 视频链接，自动完成视频内容摘要并返回结构化 Markdown 笔记。

## 功能特性

- **多平台支持**：Bilibili、YouTube 视频链接自动识别
- **批量处理**：一条消息包含多个链接时依次处理
- **智能转录**：优先提取视频自带字幕（快速路径），无字幕时回退到 Whisper AI 语音转录
- **AI 总结**：通过 OpenAI 兼容接口调用大语言模型生成中文结构化笔记
- **断点续传**：Pipeline 按步骤检查中间文件，失败后重新处理时跳过已完成步骤
- **结果缓存**：已完成的视频直接返回缓存笔记，不重复处理
- **自动重试**：单链接最多重试 3 次（指数退避）
- **自动清理**：后台定时清理过期笔记（30天）和任务文件（1天）
- **代理支持**：Telegram API 和 YouTube 下载均支持配置代理

---

## 项目结构

```
videobot/
├── bot.py          # Telegram bot 入口
├── pipeline.py     # 处理管道（URL 解析、编排、重试）
├── transcribe.py   # 字幕提取 + Whisper 语音转录（含检查点）
├── summarize.py    # OpenAI 兼容接口 LLM 总结
├── cache.py        # 笔记缓存（按视频 ID 存储）
├── cleanup.py      # 后台定时清理线程
├── config.py       # 环境变量配置
├── requirements.txt
├── .env.example
└── tests/          # 单元测试
```

---

## 前置条件

| 依赖 | 说明 |
|------|------|
| Python ≥ 3.10 | 运行时 |
| [FFmpeg](https://ffmpeg.org/download.html) | yt-dlp 音频提取所需 |
| Telegram Bot Token | [@BotFather](https://t.me/BotFather) 创建 |
| OpenAI 兼容 API Key | 支持 GPT-4o、GLM、Qwen 等任意兼容接口 |

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，至少填写 TELEGRAM_TOKEN 和 OPENAI_API_KEY
```

主要配置项：

| 变量 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `TELEGRAM_TOKEN` | ✅ | — | Bot Token |
| `OPENAI_API_KEY` | ✅ | — | LLM / Whisper API 密钥 |
| `OPENAI_API_BASE` | | `https://api.openai.com/v1` | 兼容 API 地址 |
| `OPENAI_MODEL` | | `gpt-4o` | 对话模型名称 |
| `WHISPER_MODEL` | | `whisper-1` | 语音转录模型 |
| `PROXY_URL` | | — | HTTP/SOCKS5 代理（如 `socks5://127.0.0.1:1080`） |
| `DATA_DIR` | | `data` | 数据存储根目录 |
| `NOTE_EXPIRE_DAYS` | | `30` | 笔记缓存过期天数 |
| `TASK_EXPIRE_DAYS` | | `1` | 任务文件过期天数 |
| `MAX_RETRIES` | | `3` | 单链接最大重试次数 |

### 3. 启动机器人

```bash
python bot.py
```

---

## 使用方式

向 Bot 发送包含 Bilibili 或 YouTube 链接的消息即可，支持一条消息包含多个链接：

```
看看这两个视频：
https://www.bilibili.com/video/BV1xx411c7mD
https://youtu.be/dQw4w9WgXcQ
```

Bot 会对每条链接依次回复一份结构化 Markdown 笔记：

```markdown
# 视频标题

## 核心要点
- 要点一
- 要点二
- 要点三

## 详细笔记
### 第一部分
...

## 总结
...
```

---

## 处理流程

```
消息 → URL 检测
         ↓
    缓存命中? ──是──→ 直接返回笔记
         ↓否
    字幕提取 (yt-dlp)
         ↓ 无字幕
    音频下载 → Whisper 转录
         ↓
    LLM 总结 (OpenAI 兼容接口)
         ↓
    写入缓存 → 返回笔记
```

每个中间步骤的结果保存在 `data/tasks/{video_id}/` 下，失败重试时自动跳过已完成步骤。

---

## 运行测试

```bash
python -m pytest tests/ -v
```

---

## 许可证

MIT
