"""
AI 同声传译助手 — 全链路管道

音频采集 → VAD → ASR(faster-whisper) → LLM翻译 → 修正 → WebSocket推送

启动方式:
    python main.py                    # 完整管道 + WebSocket
    python main.py --mock             # 使用 MockASR（调试）
    python main.py --no-ws            # 不启动 WebSocket（仅控制台输出）
    python main.py --list-devices     # 列出音频设备

按 Ctrl+C 优雅退出。
"""

from __future__ import annotations

import asyncio
import random
import signal
import sys
import threading
import time
from typing import NoReturn

from loguru import logger

from config import settings
from core.audio import AudioCapture, VADProcessor
from core.asr import WhisperASREngine, MockASREngine
from core.translation import TranslationEngine
from core.correction import CorrectionEngine
from core.models.schemas import ASREvent, TransEvent, CorrectionEvent


class Pipeline:
    """全链路管道管理器。"""

    def __init__(self, use_mock: bool = False, enable_ws: bool = True):
        self._use_mock = use_mock
        self._enable_ws = enable_ws

        # ── 队列 ───────────────────────────────
        self.capture_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=200)
        self.vad_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=50)
        self.asr_queue: asyncio.Queue[ASREvent] = asyncio.Queue()          # ASR输出
        self.asr_ws_queue: asyncio.Queue[ASREvent] = asyncio.Queue()       # ASR→前端（即时显示）
        self.trans_in_queue: asyncio.Queue[ASREvent] = asyncio.Queue()     # ASR→翻译
        self.trans_out_queue: asyncio.Queue[TransEvent] = asyncio.Queue()  # 翻译→修正
        self.output_queue: asyncio.Queue[TransEvent | CorrectionEvent] = asyncio.Queue()
        self.correction_status_queue: asyncio.Queue = asyncio.Queue()

        # ── 组件 ───────────────────────────────
        if not use_mock:
            self.capture = AudioCapture()
            self.capture.output_queue = self.capture_queue
        else:
            self.capture = None

        self.vad = VADProcessor(input_queue=self.capture_queue, output_queue=self.vad_queue)

        if use_mock:
            self.asr = MockASREngine(input_queue=self.vad_queue, output_queue=self.asr_queue)
        else:
            self.asr = WhisperASREngine(input_queue=self.vad_queue, output_queue=self.asr_queue)

        self.translator = TranslationEngine(input_queue=self.trans_in_queue, output_queue=self.trans_out_queue)
        self.corrector = CorrectionEngine(
            input_queue=self.trans_out_queue,
            output_queue=self.output_queue,
            status_queue=self.correction_status_queue,
        )

        self.ws_server = None
        self._uvicorn_thread: threading.Thread | None = None

    # ──── 启动 ────────────────────────────────

    async def start(self) -> None:
        """启动所有组件。"""
        # 1. 先启动 WebSocket 服务（独立线程，不影响 asyncio 队列）
        if self._enable_ws:
            self._start_websocket()

        # 2. 先启动所有下游消费者任务，确保队列被消费
        #    再启动音频采集，防止队列在消费端未就绪时爆满
        logger.info("=" * 50)
        logger.info("AI 同声传译助手 — 全链路运行中")
        logger.info("  音频    : {} Hz, {}ch, {}ms/帧",
                     settings.SAMPLE_RATE, settings.CHANNELS, settings.FRAME_DURATION_MS)
        logger.info("  VAD     : {} | 静音阈值 {}ms",
                     self.vad.backend_name, settings.SILENCE_DURATION_MS)
        logger.info("  ASR     : {} (size={}, device={})",
                     "Mock" if self._use_mock else "faster-whisper",
                     settings.ASR_MODEL_SIZE, settings.ASR_DEVICE)
        logger.info("  LLM     : {} | {}", settings.LLM_PROVIDER, settings.LLM_MODEL)
        logger.info("  修正     : {} | window={} threshold={}",
                     "开启" if settings.ENABLE_CORRECTION else "关闭",
                     settings.CORRECTION_WINDOW_SIZE, settings.CORRECTION_CONFIDENCE_THRESHOLD)
        logger.info("  WS      : {} → ws://{}:{}",
                     "启动" if self._enable_ws else "禁用",
                     settings.WS_HOST, settings.WS_PORT)
        logger.info("=" * 50)

        async with asyncio.TaskGroup() as tg:
            tg.create_task(self.vad.run(), name="VAD")
            tg.create_task(self.asr.run(), name="ASR")
            tg.create_task(self._asr_fanout(), name="ASRFanout")
            # 多路并行翻译（同时处理多个句子，大幅降低等待时间）
            tg.create_task(self.translator.run(), name="Translator-1")
            tg.create_task(self.translator.run(), name="Translator-2")
            tg.create_task(self.translator.run(), name="Translator-3")
            tg.create_task(self.corrector.run(), name="Corrector")
            if self._enable_ws and self.ws_server:
                tg.create_task(self.ws_server.run(), name="WebSocket")
            else:
                # 无 WebSocket 时用控制台输出
                tg.create_task(self._consumer(), name="Consumer")

            # 3. 给下游任务一点时间完成初始化再启动音频采集
            await asyncio.sleep(0.5)
            if self.capture is not None:
                await self.capture.start()
            else:
                # Mock 模式：启动模拟音频源
                tg.create_task(self._mock_audio_source(), name="MockAudio")

    async def stop(self) -> None:
        """停止所有组件。"""
        logger.info("正在关闭管道...")
        if self.capture is not None:
            await self.capture.stop()

    # ──── ASR 扇出（智能拼接 + 超时刷新）────────

    async def _asr_fanout(self) -> None:
        """将 ASR 结果智能拼接为完整句子后再翻译。

        策略：
        - ASR 片段立即推送前端（实时显示，用户零等待）
        - 缓冲区累积 VAD 分段结果，拼接成完整句子
        - 触发翻译条件（任一）：句末标点、超过 15 词、或 2 秒无新 ASR
        - 2 秒超时机制确保用户停顿时缓冲区一定会被刷出
        """
        import re
        self._sentence_buffer: str = ""
        self._buffer_first_event: ASREvent | None = None  # buffer 第一个 ASR 事件（固定 seq_id）
        self._buffer_latest_event: ASREvent | None = None  # buffer 最新 ASR 事件

        def _flush_buffer(reason: str) -> None:
            """刷出缓冲区送翻译。"""
            if not self._sentence_buffer or not self._buffer_first_event:
                return
            text = self._sentence_buffer
            ev = self._buffer_first_event
            self._sentence_buffer = ""
            self._buffer_first_event = None
            self._buffer_latest_event = None
            logger.info("翻译触发 ({}): [{}] \"{}\"", reason, ev.seq_id, text[:60])
            complete_event = ASREvent(
                seq_id=ev.seq_id,
                text=text,
                start_time=ev.start_time,
                end_time=ev.end_time,
                duration_ms=ev.duration_ms,
                confidence=ev.confidence,
                is_final=True,
            )
            asyncio.create_task(self.trans_in_queue.put(complete_event))

        try:
            while True:
                # 同时等待 ASR 事件和 2 秒超时
                get_task = asyncio.create_task(self.asr_queue.get())
                done, pending = await asyncio.wait(
                    [get_task],
                    timeout=2.0,
                )

                if done:
                    event = get_task.result()
                    text = event.text.strip()
                    words = text.split()

                    # 跳过过短噪声
                    if not text or len(words) < 2:
                        logger.debug("扇出跳过短文本: seq={} \"{}\"", event.seq_id, text[:30])
                        continue

                    # 累积到缓冲区
                    if self._sentence_buffer:
                        self._sentence_buffer += " " + text
                    else:
                        # 新 buffer 开始，记录第一个事件作为固定 seq_id
                        self._sentence_buffer = text
                        self._buffer_first_event = event
                    self._buffer_latest_event = event

                    # 推送最新文本到前端（使用固定 seq_id，始终更新同一条目）
                    first_ev = self._buffer_first_event
                    update_ws_event = ASREvent(
                        seq_id=first_ev.seq_id,
                        text=self._sentence_buffer,
                        start_time=first_ev.start_time,
                        end_time=event.end_time,
                        confidence=max(first_ev.confidence, event.confidence),
                        is_final=False,
                    )
                    await self.asr_ws_queue.put(update_ws_event)

                    # 检查是否完整句子：句末标点 或 超过 15 词
                    is_end = bool(re.search(r'[.!?…]\s*$', self._sentence_buffer.strip()))
                    is_long = len(self._sentence_buffer.split()) >= 15
                    if is_end or is_long:
                        _flush_buffer("句末标点" if is_end else "长句")
                else:
                    # 2 秒超时：用户暂停了，刷出缓冲区
                    get_task.cancel()
                    if self._sentence_buffer:
                        _flush_buffer("停顿超时")

        except asyncio.CancelledError:
            if self._sentence_buffer and self._buffer_first_event:
                _flush_buffer("退出")
            logger.debug("ASR 扇出任务已停止")
            raise

    # ──── WebSocket ────────────────────────────

    def _start_websocket(self) -> None:
        """在独立线程中启动 uvicorn。"""
        from core.websocket import WebSocketServer
        from core.websocket.server import start_uvicorn

        self.ws_server = WebSocketServer(
            input_queue=self.output_queue,
            asr_queue=self.asr_ws_queue,
            correction_status_queue=self.correction_status_queue,
        )

        self._uvicorn_thread = threading.Thread(
            target=start_uvicorn,
            name="uvicorn",
            daemon=True,
        )
        self._uvicorn_thread.start()
        logger.info("WebSocket 服务已启动 (ws://{}:{})", settings.WS_HOST, settings.WS_PORT)

    # ──── Mock 音频源（无麦克风时使用）───────

    async def _mock_audio_source(self) -> None:
        """模拟音频源：生成带静音间隔的语音帧。"""
        logger.info("Mock 音频源已启动（模拟语音输入）")

        import numpy as np

        sample_rate = settings.SAMPLE_RATE
        frame_bytes = settings.frame_size_bytes
        frame_samples = frame_bytes // 2
        # 预计算一个语音周期波形（200Hz 正弦波）
        cycle_len = sample_rate // 200
        base_wave = (np.sin(2 * np.pi * np.arange(cycle_len) / cycle_len) * 3000).astype(np.int16)

        def make_speech_frame(offset: int) -> bytes:
            # 从波形中采样一帧，模拟自然语音
            start = offset % cycle_len
            indices = (np.arange(frame_samples) + start) % cycle_len
            samples = base_wave[indices]
            # 加一点随机扰动
            samples = samples.astype(np.int32) + np.random.randint(-300, 300, frame_samples)
            return np.clip(samples, -32768, 32767).astype(np.int16).tobytes()

        def make_silence_frame() -> bytes:
            return b'\x00' * frame_bytes

        frame_offset = 0
        frame_duration = settings.FRAME_DURATION_MS / 1000  # ~0.02s
        try:
            while True:
                # 每轮：2~4秒语音 + 1.5~3秒静音
                speech_frames = random.randint(100, 200)  # 20ms帧 → 2~4秒
                silence_frames = random.randint(75, 150)  # 1.5~3秒

                for _ in range(speech_frames):
                    frame = make_speech_frame(frame_offset)
                    frame_offset = (frame_offset + frame_samples) % cycle_len
                    await self.capture_queue.put(frame)
                    await asyncio.sleep(frame_duration)  # 模拟实时流速

                for _ in range(silence_frames):
                    await self.capture_queue.put(make_silence_frame())
                    await asyncio.sleep(frame_duration)

                logger.debug("Mock 音频源: {:.1f}s 语音 + {:.1f}s 静音",
                             speech_frames * 0.02, silence_frames * 0.02)
        except asyncio.CancelledError:
            logger.debug("Mock 音频源已停止")
            raise

    # ──── 消费者 ───────────────────────────────

    async def _consumer(self) -> None:
        """消费 output_queue 并打印到控制台。"""
        logger.info("输出消费者已就绪...\n")

        try:
            while True:
                event = await self.output_queue.get()
                self._print_event(event)
        except asyncio.CancelledError:
            while not self.output_queue.empty():
                try:
                    self._print_event(self.output_queue.get_nowait())
                except asyncio.QueueEmpty:
                    break
            raise

    def _print_event(self, event: TransEvent | CorrectionEvent) -> None:
        """格式化输出事件。"""
        bar = "▎" + "─" * 58
        print(bar)
        if isinstance(event, TransEvent):
            print(f"  翻译 [{event.seq_id:04d}] [{event.status}]")
            print(f"  EN: {event.source_text}")
            print(f"  CN: {event.translated_text}")
        elif isinstance(event, CorrectionEvent):
            print(f"  修正 [{event.seq_id:04d}]")
            print(f"  原文: {event.original_source or 'N/A'}")
            print(f"  旧译: {event.original_translation}")
            print(f"  新译: {event.corrected_translation}")
            if event.reason:
                print(f"  原因: {event.reason}")
        print()


# ──── 入口 ──────────────────────────────────────────

def main() -> NoReturn:
    """CLI 入口。"""
    import argparse

    parser = argparse.ArgumentParser(description="AI 同声传译助手")
    parser.add_argument("--mock", action="store_true", help="使用 MockASR（调试用）")
    parser.add_argument("--no-ws", action="store_true", help="不启动 WebSocket 服务")
    parser.add_argument("--list-devices", action="store_true", help="列出音频设备")
    args = parser.parse_args()

    # 列出设备并退出
    if args.list_devices:
        AudioCapture.list_devices()
        sys.exit(0)

    # Windows asyncio 兼容性
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    pipeline = Pipeline(use_mock=args.mock, enable_ws=not args.no_ws)

    async def _amain() -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, lambda: asyncio.create_task(_shutdown(pipeline)))
            except NotImplementedError:
                pass
        await pipeline.start()

    async def _shutdown(pl: Pipeline) -> None:
        logger.info("\n正在关闭...")
        for task in asyncio.all_tasks():
            if task is not asyncio.current_task():
                task.cancel()

    try:
        asyncio.run(_amain())
    except KeyboardInterrupt:
        logger.info("用户中断")
        sys.exit(0)


if __name__ == "__main__":
    main()
