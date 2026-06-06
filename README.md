# AI 同声传译助手 🎙️

> 基于 AI 的实时同声传译工具。将英语语音实时识别并翻译为中文，以字幕形式呈现，支持自动修正。

## ✨ 功能

- **实时语音采集** — 通过麦克风采集音频，支持 VAD 语音活动检测智能切分
- **AI 语音识别** — 基于 faster-whisper（CUDA 加速），将英语语音转为文字
- **AI 实时翻译** — 调用 OpenAI 兼容 API（DeepSeek / GPT）翻译为中文
- **智能句子拼接** — 自动识别句末标点 + 超时刷新，翻译内容完整不破碎
- **自动修正引擎** — 滑动窗口审查翻译质量，发现错误自动修正并回显
- **Web 字幕界面** — 实时推送翻译结果到浏览器，支持修正痕迹展示
- **多路并行** — 3 路翻译并行处理，大幅降低等待延迟

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

> **Windows 依赖说明**：`webrtcvad` 在 Windows 上可能编译失败，系统会自动降级为 `Energy VAD`（精度略低但可用）。如遇 `pyaudio` 安装失败，请从 [这里](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio) 下载对应 wheel。

### 2. 配置 API Key

```bash
cp .env.example .env
```

编辑 `.env` 文件，填入你的 API Key：

```ini
# DeepSeek（推荐，速度快性价比高）
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-key-here
OPENAI_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat

# 或者使用 OpenAI
# OPENAI_BASE_URL=https://api.openai.com/v1
# LLM_MODEL=gpt-4o-mini
```

### 3. 启动

```bash
# 完整模式（需要麦克风）
python main.py

# Mock 模式（无需麦克风，模拟音频输入用于测试）
python main.py --mock

# 仅控制台输出（不启动 Web 界面）
python main.py --no-ws

# 列出音频设备
python main.py --list-devices
```

启动后浏览器打开 **[http://localhost:8765](http://localhost:8765)**

## 📋 数据流水线

```
麦克风 → AudioCapture → VAD 切分 → Whisper ASR → LLM 翻译
                                                       ↓
Web 前端 ← WebSocket ← 修正引擎 ← 智能拼接 ← 翻译结果
                             ↓ (发现错误)
                       修正事件 → 前端字幕更新
```

## ⚙️ 配置说明

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `ASR_MODEL_SIZE` | `tiny` | Whisper 模型大小：`tiny`/`base`/`small`/`medium`/`large` |
| `ASR_DEVICE` | `cuda` | 推理设备：`cuda` / `cpu` |
| `ASR_COMPUTE_TYPE` | `float16` | 精度：`float16` / `int8` / `float32` |
| `VAD_MODE` | `1` | VAD 激进程度：0(宽松) ~ 3(激进) |
| `SILENCE_DURATION_MS` | `600` | 语音结束静音判定 (ms) |
| `ENABLE_CORRECTION` | `true` | 是否启用滑动窗口修正 |
| `CORRECTION_WINDOW_SIZE` | `8` | 修正窗口大小 |
| `LLM_PROVIDER` | `openai` | 翻译提供商：`openai` / `anthropic` |
| `LLM_MODEL` | `deepseek-chat` | 翻译模型 |
| `WS_PORT` | `8765` | WebSocket 服务端口 |

## 🏗️ 项目结构

```
ai-realtime-translator/
├── main.py                    # CLI 入口 + 管道编排
├── config.py                  # 全局配置 (pydantic-settings)
├── requirements.txt           # Python 依赖
├── .env.example               # 环境变量模板
├── .gitignore
├── README.md
├── core/
│   ├── audio/
│   │   ├── capture.py         # 音频采集 (PyAudio)
│   │   └── vad.py             # 语音活动检测 (WebRTC / Energy)
│   ├── asr/
│   │   └── engine.py          # 语音识别 (faster-whisper)
│   ├── translation/
│   │   └── engine.py          # LLM 翻译引擎
│   ├── correction/
│   │   └── engine.py          # 滑动窗口修正引擎
│   ├── websocket/
│   │   └── server.py          # WebSocket 服务 (FastAPI)
│   └── models/
│       └── schemas.py         # 数据模型 (pydantic)
└── frontend/
    └── index.html              # Web 字幕界面
```

## 🧠 修正机制

修正引擎维护一个滑动窗口（默认 8 句），窗口中翻译积累到阈值后触发 LLM 审查：

- **发现错误** → 不删除原文，在原文下方追加绿色修正行，标记 `[需修正]`
- **翻译正确** → 静默跳过，无感知
- **并行异步** → 修正审查在后台执行，不阻塞翻译主线

## 📄 许可证

MIT
