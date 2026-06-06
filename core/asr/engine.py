"""
ASR 引擎模块

提供 WhisperASREngine —— 基于 faster-whisper 的真实语音识别引擎。
接收 VAD 切分的语音段 (PCM 16kHz 16-bit mono)，输出结构化的 ASREvent。

数据流::

    VAD 输出 (bytes)  →  WhisperASREngine  →  ASREvent (含置信度)
"""

from __future__ import annotations

import asyncio
import io
import math
import time
import wave
from typing import Optional

import numpy as np
from loguru import logger

from config import settings
from core.models.schemas import ASREvent


class WhisperASREngine:
    """基于 faster-whisper 的真实 ASR 引擎。

    用法::

        engine = WhisperASREngine(input_queue, output_queue)
        await engine.run()   # 长时间运行的任务
    """

    def __init__(
        self,
        input_queue: asyncio.Queue[bytes],
        output_queue: asyncio.Queue[ASREvent],
    ) -> None:
        self._input_queue = input_queue
        self._output_queue = output_queue

        self._seq_id: int = 0
        self._total_processed: int = 0
        self._running: bool = False
        self._model = None  # lazy init

    # ──── 属性 ──────────────────────────────────────

    @property
    def total_processed(self) -> int:
        return self._total_processed

    # ──── 延迟加载模型 ──────────────────────────────

    def _ensure_model(self):
        """懒加载 faster-whisper 模型。"""
        if self._model is not None:
            return
        from faster_whisper import WhisperModel

        logger.info(
            "正在加载 Whisper 模型 | size={} device={} compute={}",
            settings.ASR_MODEL_SIZE,
            settings.ASR_DEVICE,
            settings.ASR_COMPUTE_TYPE,
        )
        t0 = time.perf_counter()
        self._model = WhisperModel(
            settings.ASR_MODEL_SIZE,
            device=settings.ASR_DEVICE,
            compute_type=settings.ASR_COMPUTE_TYPE,
            cpu_threads=4,
            num_workers=1,
        )
        elapsed = time.perf_counter() - t0
        logger.info("Whisper 模型加载完成 ({:.1f}s)", elapsed)

    # ──── 主循环 ────────────────────────────────────

    async def run(self) -> None:
        """主循环：消费语音段 → ASR → 产生 ASREvent。"""
        self._ensure_model()
        self._running = True
        logger.info("WhisperASREngine 已启动")

        try:
            while self._running:
                audio_segment = await self._input_queue.get()
                await self._process_segment(audio_segment)
        except asyncio.CancelledError:
            logger.debug("WhisperASREngine 任务被取消")
            self._running = False
            raise

    def stop(self) -> None:
        self._running = False

    # ──── 内部处理 ──────────────────────────────────

    async def _process_segment(self, audio_segment: bytes) -> None:
        """处理单个语音段：运行 Whisper 推理。"""
        seq_id = self._seq_id
        self._seq_id += 1
        self._total_processed += 1

        segment_duration_ms = self._calc_duration_ms(audio_segment)
        start_time = time.monotonic()

        # 将 PCM bytes 转成 float32 numpy 数组
        audio_array = self._pcm_to_float32(audio_segment)

        try:
            # 在线程池中运行 Whisper 推理（阻塞操作）
            segments, info = await asyncio.get_event_loop().run_in_executor(
                None, self._transcribe, audio_array
            )

            segments = list(segments)
            elapsed_ms = (time.monotonic() - start_time) * 1000

            if not segments:
                logger.debug("ASR 未检测到语音内容")
                return

            # 拼接多个 segment 的文本
            full_text = " ".join(seg.text for seg in segments).strip()
            avg_confidence = (
                sum(seg.avg_logprob for seg in segments) / len(segments)
                if segments else 0.0
            )
            # 将 avg_logprob (通常 -1~0) 映射到 0~1 置信度
            confidence = 1.0 / (1.0 + math.exp(-avg_confidence * 2))

            event = ASREvent(
                seq_id=seq_id,
                text=full_text,
                start_time=time.monotonic() - elapsed_ms / 1000,
                end_time=time.monotonic(),
                duration_ms=round(segment_duration_ms, 1),
                confidence=round(min(confidence, 0.99), 3),
                is_final=True,
            )

            await self._output_queue.put(event)

            logger.info(
                "ASR [{:04d}] | 音频={:.0f}ms 推理={:.0f}ms 置信度={:.0%} | \"{}\"",
                seq_id, segment_duration_ms, elapsed_ms, confidence, full_text[:50],
            )

        except Exception as exc:
            logger.exception("ASR 处理异常: {}", exc)

    def _transcribe(self, audio: np.ndarray):
        """同步调用 faster-whisper transcribe，在 executor 中执行。"""
        return self._model.transcribe(
            audio,
            language="en",
            beam_size=5,
            vad_filter=False,  # 上游 VAD 已经做了切分
            condition_on_previous_text=False,
        )

    # ──── 工具方法 ──────────────────────────────────

    @staticmethod
    def _pcm_to_float32(pcm_bytes: bytes) -> np.ndarray:
        """将 PCM int16 bytes 转换为 float32 numpy 数组 (-1~1)。"""
        samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
        samples /= 32768.0
        return samples

    @staticmethod
    def _calc_duration_ms(pcm_bytes: bytes) -> float:
        num_samples = len(pcm_bytes) // settings.SAMPLE_WIDTH
        return (num_samples / settings.SAMPLE_RATE) * 1000


# ══════════════════════════════════════════════════
# 保留 MockASREngine 作为测试/调试备用
# ══════════════════════════════════════════════════

import random

_MOCK_CORPUS = [
    "Hello everyone, welcome to today's meeting.",
    "Let me share some updates on the project.",
    "We have made significant progress this quarter.",
    "The new feature has been deployed to production.",
    "I would like to discuss the next steps for our team.",
    "Thank you for your attention and participation.",
    "Please let me know if you have any questions.",
    "We will continue working on the remaining issues.",
    "Our revenue has grown by twenty percent year over year.",
    "The customer satisfaction score is at an all time high.",
]


class MockASREngine:
    """模拟 ASR 引擎（调试/演示用）。"""

    def __init__(
        self,
        input_queue: asyncio.Queue[bytes],
        output_queue: asyncio.Queue[ASREvent],
    ) -> None:
        self._input_queue = input_queue
        self._output_queue = output_queue
        self._seq_id: int = 0
        self._total_processed: int = 0
        self._running: bool = False
        self._last_output_time: float = 0.0

    @property
    def total_processed(self) -> int:
        return self._total_processed

    async def run(self) -> None:
        self._running = True
        logger.info("MockASREngine 已启动 | 模拟语料数={}", len(_MOCK_CORPUS))
        try:
            while self._running:
                audio_segment = await self._input_queue.get()
                await self._process_segment(audio_segment)
        except asyncio.CancelledError:
            logger.debug("MockASREngine 任务被取消")
            self._running = False
            raise

    def stop(self) -> None:
        self._running = False

    async def _process_segment(self, audio_segment: bytes) -> None:
        seq_id = self._seq_id
        self._seq_id += 1
        self._total_processed += 1
        segment_duration_ms = self._calc_duration_ms(audio_segment)

        process_ms = segment_duration_ms * random.uniform(0.2, 0.5) + random.random() * 100
        process_ms = min(process_ms, 2000)
        await asyncio.sleep(process_ms / 1000)

        text = _MOCK_CORPUS[seq_id % len(_MOCK_CORPUS)]
        confidence = 0.80 + random.random() * 0.18

        event = ASREvent(
            seq_id=seq_id,
            text=text,
            start_time=time.monotonic() - segment_duration_ms / 1000,
            end_time=time.monotonic(),
            duration_ms=round(segment_duration_ms, 1),
            confidence=round(confidence, 3),
            is_final=True,
        )
        await self._output_queue.put(event)
        self._last_output_time = time.monotonic()

        logger.info(
            "ASR [{:04d}] | 音频={:.0f}ms 处理={:.0f}ms 置信度={:.0%} | \"{}\"",
            seq_id, segment_duration_ms, process_ms, confidence, text[:50],
        )

    @staticmethod
    def _calc_duration_ms(pcm_bytes: bytes) -> float:
        num_samples = len(pcm_bytes) // settings.SAMPLE_WIDTH
        return (num_samples / settings.SAMPLE_RATE) * 1000
