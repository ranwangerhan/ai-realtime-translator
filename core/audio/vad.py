"""
语音活动检测 (VAD) 模块

将原始音频帧流切分为"静音段"和"语音段"。
完整语音段被送入 ASR 引擎进行识别。

支持后端（自动选择）:
  1. WebRTC VAD     — 最佳质量，需要编译 C 扩展 (pip install webrtcvad)
  2. Energy VAD     — 基于 RMS 能量的简易 VAD，零依赖

数据流::

    capture_queue (单帧)  →→  VADProcessor  →→  vad_output_queue (语音段)
"""

from __future__ import annotations

import asyncio
import math
import struct
from abc import ABC, abstractmethod
from typing import Optional

from loguru import logger

from config import settings


# ══════════════════════════════════════════════════════════
# VAD 后端抽象
# ══════════════════════════════════════════════════════════

class _VADBackend(ABC):
    """VAD 算法接口。"""

    @abstractmethod
    def is_speech(self, frame: bytes, sample_rate: int) -> bool:
        """判断一帧是否为语音。"""

    @property
    @abstractmethod
    def name(self) -> str:
        """后端名称（日志用）。"""


class _WebRtcVAD(_VADBackend):
    """WebRTC VAD 封装。"""

    def __init__(self, mode: int) -> None:
        import webrtcvad  # type: ignore[import-untyped]
        self._vad = webrtcvad.Vad(mode)

    def is_speech(self, frame: bytes, sample_rate: int) -> bool:
        return self._vad.is_speech(frame, sample_rate)

    @property
    def name(self) -> str:
        return "WebRTC VAD"


class _EnergyVAD(_VADBackend):
    """基于 RMS 能量的简易 VAD。

    适用于无编译环境的 Windows 等平台。
    通过计算 PCM 样本的均方根（RMS）并与自适应阈值比较判断语音。
    """

    # int16 的 RMS 在静音环境下通常 < 50，正常说话 > 500-3000
    # 绝对音量下限：任何低于该值的都是静音
    # 此值需高于环境底噪 RMS，否则 VAD 会把底噪当成连续语音
    # 设置原则：环境底噪RMS × 1.5，用户可在运行时微调
    _ABSOLUTE_FLOOR: float = 120.0
    # 离开语音模式时的退出门槛倍率（低于激活门槛，实现磁滞）
    # 需要保证 silence_thr > 环境底噪 RMS，否则语音段永不结束
    _HYSTERESIS_RATIO: float = 0.85

    def __init__(self, mode: int = 1) -> None:
        # 对齐 WebRTC VAD 语义：mode 越大 = 越激进 = 更难触发语音
        # 所以 mode 越大 → threshold_multiplier 越大
        self._threshold_multiplier = {0: 1.0, 1: 1.5, 2: 2.5, 3: 4.0}.get(mode, 1.5)
        self._noise_floor: Optional[float] = None
        self._frame_count = 0
        # 磁滞状态：当前是否处于语音激活模式
        self._speech_active: bool = False

    @property
    def _speech_threshold(self) -> float:
        """进入语音模式的激活阈值。"""
        nf = self._noise_floor if self._noise_floor is not None else self._ABSOLUTE_FLOOR
        return max(nf * self._threshold_multiplier, self._ABSOLUTE_FLOOR)

    @property
    def _silence_threshold(self) -> float:
        """退出语音模式的静音阈值（比激活阈值低，实现磁滞）。"""
        nf = self._noise_floor if self._noise_floor is not None else self._ABSOLUTE_FLOOR
        base = max(nf * self._threshold_multiplier, self._ABSOLUTE_FLOOR)
        return max(base * self._HYSTERESIS_RATIO, self._ABSOLUTE_FLOOR * 0.5)

    def is_speech(self, frame: bytes, sample_rate: int) -> bool:
        # 计算 RMS
        count = len(frame) // 2
        if count == 0:
            return False
        # 从 int16 小端 PCM 计算均方根
        samples = struct.unpack_from(f"<{count}h", frame)
        rms = math.sqrt(sum(s * s for s in samples) / count)

        # 自适应噪声底噪估计
        if self._noise_floor is None:
            self._noise_floor = max(rms, 1.0)
        else:
            # 平滑更新：慢升快降。只在非活跃语音期更新，避免被语音拉高
            if not self._speech_active:
                if rms < self._noise_floor:
                    self._noise_floor += (rms - self._noise_floor) * 0.15
                else:
                    self._noise_floor += (rms - self._noise_floor) * 0.002

        self._frame_count += 1

        # 磁滞判定：根据当前状态选择不同的阈值
        if self._speech_active:
            # 已处于语音模式 → 用退出门槛判断何时结束
            is_speech = rms >= self._silence_threshold
            if not is_speech:
                self._speech_active = False
        else:
            # 处于静音模式 → 用激活门槛判断何时开始
            is_speech = rms >= self._speech_threshold
            if is_speech:
                self._speech_active = True

        # 每 300 帧输出调试日志（~6s间隔）
        if self._frame_count % 300 == 0:
            logger.debug(
                "VAD: rms={:.1f} floor={:.1f} act_thr={:.1f} silence_thr={:.1f} state={}",
                rms, self._noise_floor, self._speech_threshold,
                self._silence_threshold, "SPEECH" if self._speech_active else "silence",
            )

        return is_speech

    @property
    def name(self) -> str:
        return f"Energy VAD (mult={self._threshold_multiplier})"


def _create_vad(mode: int) -> _VADBackend:
    """创建 VAD 后端，自动降级。"""
    try:
        vad = _WebRtcVAD(mode)
        logger.info("VAD 后端: {} (mode={})", vad.name, mode)
        return vad
    except ImportError:
        logger.warning("webrtcvad 未安装，使用 Energy VAD 降级方案（精度较低）")
        return _EnergyVAD(mode)


# ══════════════════════════════════════════════════════════
# VAD 处理器
# ══════════════════════════════════════════════════════════

class VADProcessor:
    """VAD 处理器。

    从 ``input_queue`` 读取原始 PCM 帧，判断语音/静音。
    当检测到连续静音超过阈值后，将累积的语音段送入 ``output_queue``。
    """

    def __init__(
        self,
        input_queue: asyncio.Queue[bytes],
        output_queue: asyncio.Queue[bytes],
    ) -> None:
        self._input_queue = input_queue
        self._output_queue = output_queue

        # ── VAD 引擎 ──
        self._vad = _create_vad(settings.VAD_MODE)
        self._vad_frame_size = settings.vad_frame_size_bytes  # bytes

        # ── 帧对齐缓冲（处理 VAD 帧与采集帧不对齐的情况）──
        self._buffer = bytearray()

        # ── 状态 ──
        self._segment = bytearray()        # 当前累积的语音段
        self._silence_frames = 0           # 连续静音帧计数
        self._in_speech: bool = False      # 是否处于语音中
        self._segment_count: int = 0       # 已输出的语音段数
        self._max_segment_bytes: int = settings.SAMPLE_RATE * 2 * 5   # 最长 5 秒保护

    # ──── 公共属性 ──────────────────────────────────

    @property
    def total_segments(self) -> int:
        return self._segment_count

    @property
    def backend_name(self) -> str:
        return self._vad.name

    # ──── 主循环 ────────────────────────────────────

    async def run(self) -> None:
        """VAD 处理主循环。

        消费 ``input_queue`` 中的音频帧，输出语音段到 ``output_queue``。
        需以 asyncio task 形式运行。
        """
        logger.info(
            "VAD 已启动 | {} | 帧={}ms 静音阈值={}帧({}ms) 最长段={}s",
            self._vad.name,
            settings.VAD_FRAME_MS,
            settings.silence_frame_threshold,
            settings.SILENCE_DURATION_MS,
            self._max_segment_bytes // (settings.SAMPLE_RATE * 2),
        )

        try:
            while True:
                frame = await self._input_queue.get()
                self._buffer.extend(frame)
                self._process_buffer()
        except asyncio.CancelledError:
            logger.debug("VAD 任务被取消")
            await self._flush_remaining()
            raise

    # ──── 内部处理 ──────────────────────────────────

    def _process_buffer(self) -> None:
        """从 buffer 中切出 VAD 帧逐一判断。"""
        while len(self._buffer) >= self._vad_frame_size:
            vad_frame = bytes(self._buffer[: self._vad_frame_size])
            self._buffer = self._buffer[self._vad_frame_size :]

            try:
                is_speech = self._vad.is_speech(vad_frame, settings.SAMPLE_RATE)
            except Exception:
                logger.opt(exception=True).warning("VAD 判断异常")
                is_speech = False

            self._handle_frame(vad_frame, is_speech)

        # 防止 buffer 因异常无限堆积
        if len(self._buffer) > self._vad_frame_size * 8:
            logger.warning("VAD 缓冲异常堆积 ({} bytes)，清空", len(self._buffer))
            self._buffer.clear()

    def _handle_frame(self, frame: bytes, is_speech: bool) -> None:
        """处理单帧 VAD 判定结果。"""
        if is_speech:
            self._segment.extend(frame)
            self._silence_frames = 0
            if not self._in_speech:
                self._in_speech = True
                logger.debug("┃ 语音段开始")

            # 超长语音段强制切分（防止无限累加）
            if len(self._segment) >= self._max_segment_bytes:
                logger.info("  └─ 语音段超长 (>{:.0f}s)，强制切分", self._max_segment_bytes / (settings.SAMPLE_RATE * 2))
                self._finalize_segment(was_forced=True)

        else:
            if self._in_speech:
                self._silence_frames += 1
                # 静音未超阈值前仍归当前段（容纳词语间停顿）
                if self._silence_frames < settings.silence_frame_threshold:
                    self._segment.extend(frame)
                else:
                    self._finalize_segment()
            # 非语音阶段 → 直接丢弃帧

    def _finalize_segment(self, was_forced: bool = False) -> None:
        """完成一个语音段，送入输出队列。"""
        if len(self._segment) < self._vad_frame_size * 10:  # 最少 0.3s 语音
            logger.debug("  └─ 丢弃过短片段 ({:.1f}s, {} bytes)", len(self._segment) / (settings.SAMPLE_RATE * 2), len(self._segment))
            self._reset_state()
            return

        segment_bytes = bytes(self._segment)
        self._segment_count += 1
        duration = len(segment_bytes) / (settings.SAMPLE_RATE * 2)

        log_msg = f"  └─ 语音段 #{self._segment_count}: {duration:.1f}s ({len(segment_bytes)} bytes)"
        if was_forced:
            log_msg += " [强制切分]"
        logger.info(log_msg)

        asyncio.ensure_future(self._safe_put(segment_bytes))
        self._reset_state()

    def _reset_state(self) -> None:
        """重置语音段累积状态。"""
        self._segment.clear()
        self._silence_frames = 0
        self._in_speech = False

    async def _safe_put(self, data: bytes) -> None:
        """带背压保护和超时的队列写入。"""
        try:
            await asyncio.wait_for(self._output_queue.put(data), timeout=5.0)
        except asyncio.TimeoutError:
            logger.error("VAD 输出队列阻塞 >5s — 丢弃语音段 ({} bytes)", len(data))

    async def _flush_remaining(self) -> None:
        """取消时将未完成的语音段发出。"""
        if len(self._segment) >= self._vad_frame_size * 2:
            logger.debug("取消时刷出剩余语音段 ({} bytes)", len(self._segment))
            await self._safe_put(bytes(self._segment))
