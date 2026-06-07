# AI 同声传译助手 🎙️

> 基于 AI 的实时同声传译工具。**浏览器采集麦克风**，通过 WebSocket 发送到服务器，实时语音识别并翻译为中文，以字幕形式呈现，支持自动修正。

## 核心特性

- **浏览器麦克风** — 使用 `MediaRecorder / AudioContext API` 获取浏览器麦克风权限，通过 WebSocket 发送 PCM 音频
- **AI 语音识别** — 基于 faster-whisper（CUDA 加速），将英语语音转为文字
- **AI 实时翻译** — 调用 OpenAI 兼容 API（DeepSeek / GPT）翻译为中文
- **智能句子拼接** — 自动识别句末标点 + 2 秒超时刷新，翻译内容完整不破碎
- **自动修正引擎** — 滑动窗口审查翻译质量，发现错误自动修正并回显（不删除原文）
- **Web 字幕界面** — 实时推送翻译结果到浏览器，支持修正痕迹展示
- **多路并行** — 3 路翻译并行处理，大幅降低等待延迟
- **无需服务器麦克风** — 云服务器也可使用，音频采集完全在浏览器端完成
## demo演示视频



## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

> **Windows 说明**：`webrtcvad` 在 Windows 上可能编译失败，系统会自动降级为 `Energy VAD`。如遇 `pyaudio` 安装失败，请从 [这里](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio) 下载对应 wheel。

### 2. 配置 API Key

```bash
cp .env.example .env
```

编辑 `.env` 文件，填入你的 API Key：

```ini
# DeepSeek（推荐）
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
# 浏览器麦克风模式（默认，推荐！服务器部署用这个）
python main.py

# 本机声卡模式（本机有物理麦克风时用）
python main.py --local-mic

# Mock 模式（模拟音频输入，调试用）
python main.py --mock

# 仅控制台输出（不启动 Web 界面）
python main.py --no-ws
```

启动后浏览器打开 **http://localhost:8765**，点击「启动麦克风」按钮授权即可使用。

## 数据流水线

```
浏览器麦克风 → getUserMedia → AudioContext → Int16 PCM
                                                        ↓
                                                WebSocket 发送
                                                        ↓
                                              VAD 语音活动检测
                                                        ↓
                                                 Whisper ASR
                                                        ↓
                                             智能句子拼接
                                                        ↓
                                               LLM 翻译 (中文)
                                                        ↓
                                              修正引擎审查
                                             ↙         ↘
                                        翻译正确      发现错误
                                           ↓              ↓
                                      推送前端        修正事件推送
                                           ↓              ↓
                                      前端字幕        前端追加修正行
```

## 架构说明

### 浏览器麦克风模式（默认）

这是推荐的部署方式，特别适合**云服务器**：

1. 用户打开网页 → 浏览器请求麦克风权限
2. 浏览器采集音频 → `AudioContext` 以 20ms 帧为单位转为 Int16 PCM
3. 每帧通过 `WebSocket.send()` 以二进制格式发送到服务器
4. 服务器接收 → VAD 切分 → Whisper ASR → DeepSeek 翻译 → 修正
5. 结果以 JSON 文本通过同一 WebSocket 推回前端显示

### 部署到服务器

```bash
# 拉取代码
git clone https://github.com/ranwangerhan/ai-realtime-translator.git
cd ai-realtime-translator

# 安装依赖
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 配置 API Key
cp .env.example .env
vim .env

# 启动（浏览器麦克风模式）
python main.py
```

浏览器访问 `http://你的服务器IP:8765`，点击「启动麦克风」即可。

## 配置说明

> 以下为代码层面的默认值。可通过 `.env` 文件覆盖。

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `ASR_MODEL_SIZE` | `base` | Whisper 模型：`tiny`/`base`/`small`/`medium`/`large` |
| `ASR_DEVICE` | `cpu` | 推理设备：`cuda` / `cpu` |
| `ASR_COMPUTE_TYPE` | `int8` | 精度：`float16` / `int8` / `float32` |
| `VAD_MODE` | `1` | VAD 激进程度：0(宽松) ~ 3(激进) |
| `SILENCE_DURATION_MS` | `800` | 语音结束静音判定 (ms) |
| `ENABLE_CORRECTION` | `true` | 是否启用滑动窗口修正 |
| `CORRECTION_WINDOW_SIZE` | `5` | 修正窗口大小 |
| `LLM_PROVIDER` | `openai` | 翻译提供商：`openai` / `anthropic` |
| `LLM_MODEL` | `gpt-4o-mini` | 翻译模型 |
| `WS_PORT` | `8765` | WebSocket 服务端口 |

## 项目结构

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
    └── index.html              # Web 字幕界面 + 浏览器麦克风采集
```

## 修正机制

修正引擎维护一个滑动窗口（默认 5 句），窗口中翻译积累到阈值后触发 LLM 审查：

- **发现错误** → 不删除原文，在原文下方追加绿色修正行，标记 `[需修正]`
- **翻译正确** → 静默跳过，无感知
- **并行异步** → 修正审查在后台执行，不阻塞翻译主线

## 许可证

MIT
