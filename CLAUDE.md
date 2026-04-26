# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

VideoBot 是一个 Telegram 机器人，接收 Bilibili/YouTube 视频链接，自动完成下载→转录→AI总结→返回 Markdown 笔记。单用户私有部署。

## 常用命令

```bash
# 安装依赖
pip install -r requirements.txt

# 运行（需先配置 .env）
python main.py

# 运行全部测试
python -m pytest tests/ -v

# 运行单个测试文件
python -m pytest tests/test_url_parser.py -v

# 集成测试（需要网络、.env、模型）
python tests/test_integration.py
```

## 架构

### 数据流：`Pipeline.process()` (core/pipeline.py)

用户发链接 → URL解析 → 任务创建/查找 → [字幕提取→跳过转录] 或 [音频下载→Whisper转录] → LLM总结 → 保存笔记 → 清理中间文件

Pipeline 按步骤检查中间文件（audio/subtitle/transcript）实现断点续传，失败任务重新处理时跳过已完成步骤。

### 任务状态机 (core/task_manager.py)

`PENDING → DOWNLOADING → TRANSCRIBING → SUMMARIZING → COMPLETED`（可进入 `FAILED`）

- 任务 ID = URL 的 MD5 哈希
- 每个任务在 `data/tasks/{task_id}/` 下保存 status.json 和中间文件
- 完成后中间文件被清理，仅保留 status.json 和最终笔记 `data/notes/{task_id}.md`
- 重复 URL 直接返回缓存的笔记文件

### 关键模块

| 模块 | 职责 |
|------|------|
| `bot/handler.py` | Telegram 消息处理（aiogram），多链接批量处理，单链接最多重试3次 |
| `core/url_parser.py` | URL 解析和平台识别（bilibili.com/b23.tv → bilibili, youtube.com/youtu.be → youtube） |
| `downloaders/` | 统一接口 `BaseDownloader`，Bilibili/YouTube 各一个实现，基于 yt-dlp |
| `transcriber/` | `BaseTranscriber` → `WhisperTranscriber`（faster-whisper），超长音频自动分段 |
| `summarizer/llm.py` | OpenAI 兼容接口调用，默认 qwen-plus，中文笔记输出 |
| `utils/cleanup.py` | 后台定时清理：笔记30天过期，任务目录1天过期 |

### 配置 (config.py)

所有配置通过 `.env` 环境变量加载，分为 `BotConfig`、`DownloaderConfig`、`AIConfig`、`WhisperConfig`、`DataConfig` 五个类。启动时自动检测/下载 deno（YouTube 下载需要）。

### Cookie 机制

下载器 Cookie 优先级：环境变量字符串 > 项目根目录 `cookie_*.txt` 文件。YouTube 下载有4种策略回退（带/不带 cookie × 音频优先/合并提取）。

### 模型加载

Whisper 模型优先从本地 `models/whisper-{size}/` 加载，不存在则从 ModelScope 下载。懒加载（首次使用时初始化）。
