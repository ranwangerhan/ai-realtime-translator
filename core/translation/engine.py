"""
LLM 翻译引擎

将 ASR 识别出的英文文本实时翻译为中文。
使用 OpenAI / Anthropic API，支持流式输出和上下文窗口。

数据流::

    ASREvent (英文)  →  TranslationEngine  →  TransEvent (中文)
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Optional

from loguru import logger

from config import settings
from core.models.schemas import ASREvent, TransEvent

# ──── 系统提示词 ─────────────────────────────────────

SYSTEM_PROMPT = "英文→中文，只输出结果。保留专有名词。"


class TranslationEngine:
    """基于 LLM 的实时翻译引擎。

    用法::

        engine = TranslationEngine(asr_queue, trans_queue)
        await engine.run()
    """

    def __init__(
        self,
        input_queue: asyncio.Queue[ASREvent],
        output_queue: asyncio.Queue[TransEvent],
    ) -> None:
        self._input_queue = input_queue
        self._output_queue = output_queue

        self._seq_id: int = 0
        self._context: list[dict] = []  # 对话上下文窗口
        self._running: bool = False
        self._client = None  # lazy init

    # ──── 属性 ──────────────────────────────────────

    @property
    def total_translated(self) -> int:
        return self._seq_id

    # ──── 延迟加载客户端 ────────────────────────────

    def _ensure_client(self):
        """懒加载 LLM 客户端。"""
        if self._client is not None:
            return

        if settings.LLM_PROVIDER == "openai":
            from openai import AsyncOpenAI

            if not settings.OPENAI_API_KEY:
                raise ValueError(
                    "OPENAI_API_KEY 未设置，请在 .env 中配置"
                )
            self._client = AsyncOpenAI(
                api_key=settings.OPENAI_API_KEY,
                base_url=settings.OPENAI_BASE_URL or None,
            )
            logger.info(
                "LLM 翻译引擎已初始化 | provider=openai model={}",
                settings.LLM_MODEL,
            )

        elif settings.LLM_PROVIDER == "anthropic":
            try:
                from anthropic import AsyncAnthropic
            except ImportError:
                raise ImportError("使用 Anthropic 需要安装: pip install anthropic")

            if not settings.ANTHROPIC_API_KEY:
                raise ValueError(
                    "ANTHROPIC_API_KEY 未设置，请在 .env 中配置"
                )
            self._client = AsyncAnthropic(
                api_key=settings.ANTHROPIC_API_KEY,
            )
            logger.info(
                "LLM 翻译引擎已初始化 | provider=anthropic model={}",
                settings.LLM_MODEL,
            )
        else:
            raise ValueError(f"不支持的 LLM 提供商: {settings.LLM_PROVIDER}")

    # ──── 主循环 ────────────────────────────────────

    async def run(self) -> None:
        """主循环：消费 ASR 结果 → 翻译 → 产生 TransEvent。"""
        self._ensure_client()
        self._running = True
        logger.info("TranslationEngine 已启动")

        try:
            while self._running:
                asr_event = await self._input_queue.get()
                await self._translate(asr_event)
        except asyncio.CancelledError:
            logger.debug("TranslationEngine 任务被取消")
            self._running = False
            raise

    def stop(self) -> None:
        self._running = False

    # ──── 翻译逻辑 ──────────────────────────────────

    async def _translate(self, asr_event: ASREvent) -> None:
        """调用 LLM 翻译单句。"""
        # 使用 ASR 事件的 seq_id，确保前端能匹配到对应的原文条目
        seq_id = asr_event.seq_id
        self._seq_id = max(self._seq_id, seq_id + 1)

        source_text = asr_event.text.strip()
        if not source_text:
            return

        start_time = time.perf_counter()

        try:
            translated = await self._call_llm(source_text)
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            trans_event = TransEvent(
                seq_id=seq_id,
                source_text=source_text,
                translated_text=translated,
                status="final",
                timestamp=time.time(),
            )

            await self._output_queue.put(trans_event)

            logger.info(
                "翻译 [{:04d}] | {:.0f}ms | \"{}\" → \"{}\"",
                seq_id, elapsed_ms,
                source_text[:40], translated[:40],
            )

            # 更新上下文（保留最近 3 轮）
            self._context.append(
                {"role": "user", "content": source_text}
            )
            self._context.append(
                {"role": "assistant", "content": translated}
            )
            if len(self._context) > 6:  # 3 轮对话
                self._context = self._context[-6:]

        except Exception as exc:
            logger.exception("翻译异常: {}", exc)
            # 出错时原样输出原文（降级行为）
            fallback = TransEvent(
                seq_id=seq_id,
                source_text=source_text,
                translated_text=f"[翻译失败] {source_text}",
                status="final",
                timestamp=time.time(),
            )
            await self._output_queue.put(fallback)

    async def _call_llm(self, text: str) -> str:
        """调用 LLM API 进行翻译。"""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ]

        if settings.LLM_PROVIDER == "openai":
            response = await self._client.chat.completions.create(
                model=settings.LLM_MODEL,
                messages=messages,
                temperature=0,
                max_tokens=128,
            )
            return response.choices[0].message.content.strip()

        elif settings.LLM_PROVIDER == "anthropic":
            response = await self._client.messages.create(
                model=settings.LLM_MODEL,
                system=SYSTEM_PROMPT,
                messages=messages[1:],  # anthropic 的 system 单独传
                temperature=0,
                max_tokens=128,
            )
            return response.content[0].text.strip()

        raise ValueError(f"不支持的 LLM 提供商: {settings.LLM_PROVIDER}")
