"""
音频采集模块

使用 PyAudio 实现非阻塞回调式音频流读取。
采集到的原始 PCM 帧通过 asyncio.Queue 送给下游 VAD 模块。

**线程安全说明**：PyAudio 的回调在独立音频线程中执行，
通过 loop.call_soon_threadsafe 桥接到 asyncio 事件循环。
"""

from __future__ import annotations

import asyncio
import importlib
from typing import Optional

from loguru import logger

from config import settings


def _get_pyaudio():
    """延迟导入 pyaudio（避免未安装时模块级 import 失败）。"""
    try:
        return importlib.import_module("pyaudio")
    except ImportError:
        raise ImportError(
            "缺少 pyaudio 库。请执行: pip install pyaudio\n"
            "Windows 用户可从 https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio 下载 wheel"
        ) from None


class AudioCapture:
    """非阻塞音频采集器。

    用法::

        capture = AudioCapture()
        await capture.start()
        # 在另一个 task 中消费 capture.output_queue
        frame = await capture.output_queue.get()
        ...
        await capture.stop()
    """

    def __init__(self) -> None:
        self._pa = _get_pyaudio()  # 延迟导入
        self._audio: Optional["pyaudio.PyAudio"] = None
        self._stream: Optional["pyaudio.Stream"] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # 下游消费此队列获取原始 PCM 帧 (bytes)
        self.output_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=200)

        self._device_name: str = "default"

    # ──── 生命周期 ──────────────────────────────────

    async def start(self) -> None:
        """打开音频流并启动回调。"""
        self._loop = asyncio.get_running_loop()
        pa = self._pa

        self._audio = pa.PyAudio()

        chunk = settings.frame_size_bytes // settings.SAMPLE_WIDTH
        # 对 PyAudio 来说 frames_per_buffer 是采样点数
        frames_per_buffer = chunk

        try:
            self._stream = self._audio.open(
                format=pa.paInt16,
                channels=settings.CHANNELS,
                rate=settings.SAMPLE_RATE,
                input=True,
                frames_per_buffer=frames_per_buffer,
                stream_callback=self._pyaudio_callback,
                input_device_index=settings.AUDIO_DEVICE_INDEX,
            )
        except OSError as exc:
            self._audio.terminate()
            raise RuntimeError(
                f"无法打开音频输入设备 (index={settings.AUDIO_DEVICE_INDEX})。"
                f"请检查麦克风是否可用。详情: {exc}"
            ) from exc

        self._stream.start_stream()

        # 记录设备名
        if settings.AUDIO_DEVICE_INDEX is not None:
            info = self._audio.get_device_info_by_index(settings.AUDIO_DEVICE_INDEX)
            self._device_name = info.get("name", f"index={settings.AUDIO_DEVICE_INDEX}")

        logger.info(
            "音频采集已启动 | 设备={} | {}Hz {}ch {:d}-bit | 帧={}ms",
            self._device_name,
            settings.SAMPLE_RATE,
            settings.CHANNELS,
            settings.SAMPLE_WIDTH * 8,
            settings.FRAME_DURATION_MS,
        )

    async def stop(self) -> None:
        """停止并释放资源。"""
        if self._stream:
            if self._stream.is_active():
                self._stream.stop_stream()
            self._stream.close()
            self._stream = None
            logger.debug("音频流已关闭")

        if self._audio:
            self._audio.terminate()
            self._audio = None
            logger.info("PyAudio 已释放")

    # ──── 内部 ──────────────────────────────────────

    def _pyaudio_callback(
        self, in_data: bytes, frame_count: int, time_info: dict, status: int
    ) -> tuple[Optional[bytes], int]:
        """PyAudio 回调 —— 在**音频线程**中执行，必须尽快返回。"""
        if status:
            logger.warning(f"PyAudio status flag: {status}")

        if in_data is not None and self._loop is not None and not self._loop.is_closed():
            # 桥接到事件循环线程
            self._loop.call_soon_threadsafe(self._feed_queue, in_data)

        return (None, self._pa.paContinue)

    def _feed_queue(self, frame: bytes) -> None:
        """在事件循环线程中向异步队列投递一帧音频。"""
        try:
            self.output_queue.put_nowait(frame)
        except asyncio.QueueFull:
            logger.warning("音频采集队列满 (200) — 丢弃帧，系统可能过载")

    # ──── 辅助方法 ──────────────────────────────────

    @staticmethod
    def list_devices() -> None:
        """列出所有可用音频输入设备（用于排查问题）。"""
        try:
            pa = _get_pyaudio()
        except ImportError as e:
            logger.error(e)
            return
        audio = pa.PyAudio()
        for i in range(audio.get_device_count()):
            info = audio.get_device_info_by_index(i)
            if info.get("maxInputChannels", 0) > 0:
                logger.info(
                    "Device {}: {} ({} channels, {} Hz)",
                    i,
                    info.get("name"),
                    info.get("maxInputChannels"),
                    int(info.get("defaultSampleRate", 0)),
                )
        audio.terminate()
