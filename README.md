# VideoBot

一个 Telegram 机器人，接收 Bilibili / YouTube 视频链接，自动完成视频内容摘要并返回结构化 Markdown 笔记。

## 功能特性

- **多平台支持**：Bilibili、YouTube 视频链接自动识别
- **批量处理**：一条消息包含多个链接时依次处理
- **智能转录**：优先提取视频自带字幕（快速路径），无字幕时回退到 Whisper AI 语音转录
- **AI 总结**：通过 OpenAI 兼容接口调用大语言模型生成中文结构化笔记
- **断点续传**：Pipeline 按步骤检查中间文件，失败后重新处理时跳过已完成步骤
- **结果缓存**：已完成的视频直接返回缓存笔记，不重复处理
- **自动重试**：单链接最多重试 3 次
- **自动清理**：后台定时清理过期笔记（30天）和任务文件（1天）
- **代理支持**：Telegram API 和 YouTube 下载均支持配置代理

## 处理流程

```
用户发送视频链接
  → URL 解析与平台识别
  → 任务创建（或命中缓存直接返回）
  → 尝试提取视频自带字幕（快速路径）
  → 无字幕时：下载音频 → Whisper 转录
  → AI 总结生成 Markdown 笔记
  → 发送笔记文件给用户
```

## 系统依赖

| 依赖 | 用途 | 安装方式 |
|------|------|----------|
| Python 3.10+ | 运行环境 | `apt install python3` 或 conda |
| FFmpeg | 音频转码/提取 | `apt install ffmpeg` |
| Deno | yt-dlp JS challenge 解密（YouTube 下载需要） | 自动安装（首次启动时下载到 `bin/deno`） |

## 安装部署

### 1. 克隆项目

```bash
git clone <repo-url>
cd videobot
```

### 2. 安装 Python 依赖

```bash
pip install -r requirements.txt
```

依赖说明：

- `aiogram>=3.0` — Telegram Bot 框架
- `python-dotenv` — .env 环境变量加载
- `yt-dlp` — 视频下载
- `faster-whisper` — 音频转文字
- `ffmpeg-python` — 音频处理
- `openai` — AI 总结接口（OpenAI 兼容）
- `aiohttp` / `aiohttp-socks` — 代理支持

### 3. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env` 填写以下配置：

#### 必填项

| 变量 | 说明 | 示例 |
|------|------|------|
| `BOT_TOKEN` | Telegram Bot Token（从 @BotFather 获取） | `123456:ABC-DEF` |
| `AUTH_USER_ID` | 授权用户 Telegram ID（仅此用户可使用） | `7658583926` |
| `AI_API_KEY` | AI 模型 API 密钥 | `sk-xxxxxxxx` |
| `AI_API_URL` | AI 模型 API 地址（OpenAI 兼容接口） | `https://dashscope.aliyuncs.com/compatible-mode/v1` |

#### 可选项

| 变量 | 说明 | 默认值 | 示例 |
|------|------|--------|------|
| `AI_MODEL_NAME` | AI 模型名称 | `qwen-plus` | `qwen-plus` / `deepseek-chat` |
| `BOT_PROXY` | Telegram Bot 代理 | 无 | `socks5://192.168.3.212:1080` |
| `YT_PROXY` | YouTube 下载代理 | 无 | `http://192.168.3.150:7897` |
| `BILIBILI_COOKIE` | Bilibili Cookie 字符串 | 无 | 留空则使用 `cookie_bilibili.txt` |
| `YOUTUBE_COOKIE` | YouTube Cookie 字符串 | 无 | 留空则使用 `cookie_ytb.txt` |
| `WHISPER_MODEL_SIZE` | Whisper 模型大小 | `base` | `tiny` / `base` / `small` / `medium` |
| `DATA_DIR` | 数据存储目录 | `./data` | `./data` |

### 4. Cookie 文件（可选但推荐）

YouTube 和 Bilibili 部分视频需要登录 Cookie 才能下载：

- `cookie_bilibili.txt` — Bilibili cookie（Netscape 格式）
- `cookie_ytb.txt` — YouTube cookie（Netscape 格式）

获取方式（Chrome 浏览器）：
1. 安装扩展 `Get cookies.txt LOCALLY`
2. 登录对应网站
3. 导出 cookie 为 Netscape 格式，保存到项目根目录

### 5. 启动

```bash
python main.py
```

启动成功后会看到：
```
VideoBot starting...
Run polling for bot @yourbotname
```

## 使用方式

在 Telegram 中向 bot 发送视频链接：

- Bilibili: `https://www.bilibili.com/video/BV1xxxxx`
- YouTube: `https://www.youtube.com/watch?v=xxxxx`
- 短链接: `https://b23.tv/xxxxx`、`https://youtu.be/xxxxx`
- 批量：一条消息中发送多个链接，依次处理

Bot 自动完成：下载 → 转录 → AI 总结 → 返回 Markdown 笔记文件。

## 测试

```bash
# 单元测试
python -m pytest tests/ -v

# 单个测试
python -m pytest tests/test_url_parser.py -v

# 集成测试（需要网络、.env 配置、Whisper 模型）
python tests/test_integration.py
```

## 目录结构

```
videobot/
├── main.py              # 入口
├── config.py            # 配置加载（自动检测 deno）
├── bot/                 # Telegram Bot 处理
│   ├── handler.py       # 消息处理与重试逻辑
│   └── formatter.py     # Markdown 构建与临时文件
├── core/                # 核心流水线
│   ├── pipeline.py      # 主处理流程（下载→转录→总结）
│   ├── task_manager.py  # 任务状态管理与持久化
│   └── url_parser.py    # URL 解析与平台识别
├── downloaders/         # 视频下载器
│   ├── base.py          # 下载器抽象基类
│   ├── bilibili.py      # Bilibili 下载（yt-dlp）
│   └── youtube.py       # YouTube 下载（多策略回退）
├── transcriber/         # 音频转录
│   ├── base.py          # 转录器抽象基类
│   └── whisper.py       # Whisper 转录（自动分段）
├── summarizer/          # AI 总结
│   └── llm.py           # OpenAI 兼容接口调用
├── utils/               # 工具模块
│   ├── audio_processor.py # 音频预处理
│   └── cleanup.py       # 过期文件清理
├── tests/               # 测试
├── data/                # 运行时数据（git 忽略）
│   ├── notes/           # 生成的笔记
│   └── tasks/           # 任务中间文件
└── models/              # Whisper 模型（git 忽略）
```
