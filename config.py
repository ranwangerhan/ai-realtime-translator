"""
全局配置模块

使用 pydantic-settings 从环境变量 / .env 文件加载配置。
所有魔法数字集中在此外，业务代码只引用 self.config.xxx。
"""

from typing import Optional, Any
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from loguru import logger


class Settings(BaseSettings):
    """应用全局配置，字段注释即为默认值。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=False,  # 允许单测覆盖
    )

    # ──── 日志 ────
    LOG_LEVEL: str = "DEBUG"

    # ──── 音频采集 ────
    SAMPLE_RATE: int = 16000          # Hz
    CHANNELS: int = 1                 # 单声道
    SAMPLE_WIDTH: int = 2             # 16-bit = 2 bytes
    FRAME_DURATION_MS: int = 20       # 每帧时长 (ms)
    AUDIO_DEVICE_INDEX: Optional[int] = None  # None = 默认设备

    # ──── VAD ────
    VAD_MODE: int = 1                 # 0(宽松) ~ 3(激进)
    VAD_FRAME_MS: int = 30            # 10/20/30 ms —— webrtcvad 要求
    SILENCE_DURATION_MS: int = 800    # 判定一句话结束的静音阈值

    # ──── ASR ────
    ASR_MODEL_SIZE: str = "base"      # tiny / base / small / medium / large
    ASR_DEVICE: str = "cpu"           # cpu / cuda
    ASR_COMPUTE_TYPE: str = "int8"    # int8 / float16 / float32
    ASR_MODEL_PATH: Optional[str] = None  # 覆盖默认模型路径

    # ──── LLM 翻译 ────
    LLM_PROVIDER: str = "openai"      # openai / anthropic / custom
    OPENAI_API_KEY: str = ""
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"
    ANTHROPIC_API_KEY: str = ""
    LLM_MODEL: str = "gpt-4o-mini"
    LLM_TEMPERATURE: float = 0.3

    # ──── 滑动窗口修正 ────
    ENABLE_CORRECTION: bool = True
    CORRECTION_WINDOW_SIZE: int = 5
    CORRECTION_CONFIDENCE_THRESHOLD: float = 0.75

    # ──── WebSocket 服务 ────
    WS_HOST: str = "0.0.0.0"
    WS_PORT: int = 8765

    # ──── 校验器 ────

    @field_validator("AUDIO_DEVICE_INDEX", mode="before")
    @classmethod
    def _empty_str_to_none(cls, v: Any) -> Any:
        """将 .env 中的空字符串转为 None，避免 Optional[int] 校验失败。"""
        if v == "" or v is None:
            return None
        return int(v)

    # ──── 计算属性 ────

    @property
    def frame_size_bytes(self) -> int:
        """每帧字节数 = 采样率 / 1000 × 帧时长(ms) × 字节数/采样 × 声道数"""
        return int(self.SAMPLE_RATE * self.FRAME_DURATION_MS / 1000) * self.SAMPLE_WIDTH * self.CHANNELS

    @property
    def vad_frame_size_bytes(self) -> int:
        """VAD 处理的帧字节数 (10/20/30ms)"""
        return int(self.SAMPLE_RATE * self.VAD_FRAME_MS / 1000) * self.SAMPLE_WIDTH * self.CHANNELS

    @property
    def silence_frame_threshold(self) -> int:
        """判定语音结束需要的连续静音帧数"""
        return self.SILENCE_DURATION_MS // self.VAD_FRAME_MS

    def validate(self) -> "Settings":
        """启动时校验关键配置，尽早暴露错误。"""
        if self.VAD_FRAME_MS not in (10, 20, 30):
            raise ValueError(f"VAD_FRAME_MS 必须为 10/20/30，收到 {self.VAD_FRAME_MS}")
        if self.SAMPLE_RATE not in (8000, 16000, 32000, 44100, 48000):
            logger.warning(f"SAMPLE_RATE={self.SAMPLE_RATE} 可能不被所有设备支持")
        if self.LLM_PROVIDER == "openai" and not self.OPENAI_API_KEY:
            logger.warning("OPENAI_API_KEY 未设置 — LLM 翻译将在后续阶段生效")
        if self.LLM_PROVIDER == "anthropic" and not self.ANTHROPIC_API_KEY:
            logger.warning("ANTHROPIC_API_KEY 未设置 — LLM 翻译将在后续阶段生效")
        return self


# ──── 模块级单例，方便其他地方直接引用 ────
settings = Settings().validate()

# 让 loguru 适配配置级别
logger.remove()
logger.add(
    sink=lambda msg: print(msg, end=""),
    level=settings.LOG_LEVEL,
    format="<green>{time:HH:mm:ss.SSS}</green> | <level>{level:7}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    colorize=True,
)
# 同时写入文件，确保 nohup 模式下能保留日志
logger.add(
    sink="logs/pipeline.log",
    level="DEBUG",
    format="{time:HH:mm:ss.SSS} | {level:7} | {name}:{function}:{line} - {message}",
    rotation="10 MB",
    retention=3,
    enqueue=True,
)
