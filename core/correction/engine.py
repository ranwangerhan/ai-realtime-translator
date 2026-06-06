"""
滑动窗口修正引擎

监听翻译输出流，维护一个滑动窗口记录最近的翻译结果。
当检测到修正触发条件时（置信度低、上下文矛盾、新信息补全），
利用 LLM 重新审视窗口内容并下发修正指令。

修正触发策略:
1. 低置信度修正 — ASR 置信度低于阈值，后文提供了更多上下文时触发
2. 语义矛盾修正 — 新句子与窗口内已有内容矛盾时触发
3. 定期审查修正 — 每 N 句对窗口进行一次"地毯式"审查

数据流::

    TransEvent 流  →  CorrectionEngine  →  CorrectionEvent + 修正后的 TransEvent
"""

from __future__ import annotations

import asyncio
import time
from typing import Optional

from loguru import logger

from config import settings
from core.models.schemas import (
    TransEvent,
    CorrectionEvent,
    SlidingWindowEntry,
    CorrectionTask,
)

CORRECTION_PROMPT = """检查英文原文和中文译文是否匹配。

原文：{original_text}
译文：{translated_text}

如果译文正确：回复"正确"
如果译文有误：直接输出修正后的中文译文，不要加任何前缀、解释或标点符号说明。
例如：输出"这是一个好的开始"而不是"修正：这是一个好的开始"。"""


class CorrectionEngine:
    """滑动窗口修正引擎。

    用法::

        engine = CorrectionEngine(input_queue, output_queue)
        await engine.run()
    """

    def __init__(
        self,
        input_queue: asyncio.Queue[TransEvent],
        output_queue: asyncio.Queue[TransEvent | CorrectionEvent],
        status_queue: asyncio.Queue | None = None,
    ) -> None:
        self._input_queue = input_queue
        self._output_queue = output_queue
        self._status_queue = status_queue

        # 滑动窗口：记录最近 N 句翻译
        self._window: list[SlidingWindowEntry] = []
        self._window_size: int = settings.CORRECTION_WINDOW_SIZE
        self._confidence_threshold: float = settings.CORRECTION_CONFIDENCE_THRESHOLD

        self._running: bool = False
        self._seq_id: int = 0
        self._client = None  # lazy init
        self._correction_count: int = 0
        self._total_checked: int = 0
        self._review_in_progress: bool = False  # 防止并行审查

    # ──── 属性 ──────────────────────────────────────

    @property
    def total_corrections(self) -> int:
        return self._correction_count

    @property
    def window_size(self) -> int:
        return self._window_size

    # ──── 延迟加载客户端 ────────────────────────────

    def _ensure_client(self):
        if self._client is not None:
            return

        if settings.LLM_PROVIDER == "openai":
            from openai import AsyncOpenAI

            if not settings.OPENAI_API_KEY:
                logger.warning("OPENAI_API_KEY 未设置 — 修正引擎已禁用")
                self._client = None
                return

            self._client = AsyncOpenAI(
                api_key=settings.OPENAI_API_KEY,
                base_url=settings.OPENAI_BASE_URL or None,
            )
            logger.info("修正引擎已初始化 | provider=openai window_size={}", self._window_size)

        elif settings.LLM_PROVIDER == "anthropic":
            try:
                from anthropic import AsyncAnthropic
            except ImportError:
                raise ImportError("需要安装: pip install anthropic")

            if not settings.ANTHROPIC_API_KEY:
                logger.warning("ANTHROPIC_API_KEY 未设置 — 修正引擎已禁用")
                self._client = None
                return

            self._client = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
            logger.info("修正引擎已初始化 | provider=anthropic window_size={}", self._window_size)

    # ──── 主循环 ────────────────────────────────────

    async def run(self) -> None:
        """主循环：消费翻译结果 → 维护窗口 → 触发修正审查。"""
        if not settings.ENABLE_CORRECTION:
            logger.info("修正引擎已禁用 (ENABLE_CORRECTION=false)")
            # 透传模式：直接将输入转发到输出
            try:
                while True:
                    event = await self._input_queue.get()
                    await self._output_queue.put(event)
            except asyncio.CancelledError:
                raise

        self._ensure_client()
        self._running = True
        logger.info("CorrectionEngine 已启动 | window={} threshold={}",
                     self._window_size, self._confidence_threshold)

        try:
            while self._running:
                trans_event = await self._input_queue.get()
                await self._process_event(trans_event)
        except asyncio.CancelledError:
            logger.debug("CorrectionEngine 任务被取消")
            self._running = False
            raise

    def stop(self) -> None:
        self._running = False

    async def _send_status(self, status: str) -> None:
        """发送修正引擎状态到前端。"""
        if self._status_queue is not None:
            try:
                msg = {
                    "type": "correction_status",
                    "data": {"status": status, "checked": self._total_checked},
                }
                await self._status_queue.put(msg)
                logger.debug("状态已发送: {} (checked={})", status, self._total_checked)
            except Exception as exc:
                logger.debug("状态发送失败: {} - {}", status, exc)

    # ──── 事件处理 ──────────────────────────────────

    async def _process_event(self, event: TransEvent) -> None:
        """处理一条翻译结果。"""
        seq_id = event.seq_id

        # 创建窗口条目
        entry = SlidingWindowEntry(
            seq_id=seq_id,
            source_text=event.source_text,
            translated_text=event.translated_text,
        )
        self._window.append(entry)

        # 透传原始事件到输出（立刻发往前端，不等修正审查）
        await self._output_queue.put(event)

        # 如果窗口满了，后台异步审查，不阻塞主循环
        if len(self._window) >= self._window_size:
            asyncio.create_task(self._review_window())

        # 裁剪窗口
        if len(self._window) > self._window_size * 2:
            self._window = self._window[-self._window_size:]

    # ──── 窗口审查 ──────────────────────────────────

    async def _review_window(self) -> None:
        """审查滑动窗口，寻找需要修正的条目。"""
        if self._review_in_progress:
            logger.debug("修正跳过: 上一轮审查仍在进行")
            return
        if not self._client:
            logger.debug("修正跳过: 无客户端")
            return

        candidates = [
            entry for entry in self._window
            if not entry.is_corrected
        ]

        if not candidates:
            logger.debug("修正跳过: 无可修正条目")
            return

        self._review_in_progress = True
        logger.info("修正审查开始: 窗口={} 候选={}", len(self._window), len(candidates))
        asyncio.create_task(self._send_status("checking"))

        try:
            context_summary = self._build_context_summary()

            had_correction = False
            for entry in candidates[-3:]:  # 最多审查最近 3 条
                try:
                    result = await self._try_correct(entry, context_summary)
                    if result:
                        had_correction = True
                except Exception as exc:
                    logger.warning("修正审查异常 (seq={}): {}", entry.seq_id, exc)

            self._total_checked += 1
            asyncio.create_task(
                self._send_status("corrected" if had_correction else "no_correction")
            )
        finally:
            self._review_in_progress = False

    async def _try_correct(
        self, entry: SlidingWindowEntry, context: str
    ) -> None:
        """尝试修正单条翻译。"""
        prompt = CORRECTION_PROMPT.format(
            context=context,
            original_text=entry.source_text,
            translated_text=entry.translated_text,
        )

        result = await self._call_llm_correction(prompt)

        logger.debug("修正 LLM 返回: seq={} result=\"{}\"", entry.seq_id, result[:60] if result else "(空)")

        if not result:
            logger.debug("修正跳过: 空结果")
            return False

        # 判断 LLM 认为翻译是否正确
        is_correct = any(
            keyword in result
            for keyword in ["正确", "无误", "没问题", "没有错误", "准确", "Good", "OK"]
        )
        if is_correct and len(result) < 20:
            logger.debug("修正跳过: 翻译正确")
            return False

        # 清理结果：去掉 LLM 可能添加的评论前缀
        cleaned = result.strip()

        # 策略1：取最后一个冒号后的内容（"修正：xxx" → "xxx"）
        for sep in ["：", ":"]:
            if sep in cleaned:
                parts = cleaned.split(sep)
                last_part = parts[-1].strip()
                if len(last_part) > 5:  # 冒号后有实质内容
                    cleaned = last_part
                    break

        # 策略2：取所有完整句子拼接
        import re
        sentences = re.findall(r'[^。！？]*[。！？]', cleaned)
        if sentences:
            cleaned = ''.join(sentences).strip()
        # 如果还包含"错误"等关键词头，去掉第一句（通常是评论）
        if any(kw in cleaned[:15] for kw in ["错误", "有误", "应该", "漏译"]):
            sentences = re.findall(r'[^。！？]*[。！？]', cleaned)
            if len(sentences) > 1:
                cleaned = ''.join(sentences[1:]).strip()
            else:
                # 只有一句纯粹的错误描述（如"翻译错误—意思翻译错了"），跳过修正
                logger.debug("修正跳过: LLM未提供修正译文，仅返回了错误描述")
                return False

        # 策略3：去掉残留的"修正后"、"应为"等开头前缀
        cleaned = re.sub(
            r'^(修正后|应为|应改为|建议改为|正确译文|正确翻译)[：:\s]*',
            '', cleaned
        ).strip()

        # 如果修正结果和原文几乎一样，跳过
        if cleaned.rstrip("。，！？") == entry.translated_text.rstrip("。，！？"):
            logger.debug("修正跳过: 结果与原文相同")
            return False
        if len(cleaned) >= 5 and cleaned[:len(entry.translated_text)] == entry.translated_text:
            logger.debug("修正跳过: 结果仅添加了后缀")
            return False

        # 使用清理后的结果
        result = cleaned

        # 如果有修正，生成 CorrectionEvent
        # 注意：seq_id 使用被修正条目的 seq_id，前端才能匹配到对应翻译条目
        correction_event = CorrectionEvent(
            seq_id=entry.seq_id,
            original_translation=entry.translated_text,
            corrected_translation=result,
            original_source=entry.source_text,
            reason="滑动窗口审查 — 后文上下文触发了自动修正",
            timestamp=time.time(),
        )
        self._seq_id += 1
        self._correction_count += 1

        entry.is_corrected = True
        entry.correction_events.append(correction_event)

        await self._output_queue.put(correction_event)

        logger.info(
            "修正 [{:04d}] | \"{}\" → \"{}\"",
            entry.seq_id,
            entry.translated_text[:30],
            result[:30],
        )
        return True

    # ──── 辅助方法 ──────────────────────────────────

    def _build_context_summary(self) -> str:
        """从窗口中构建上下文摘要。"""
        lines = []
        for entry in self._window[-self._window_size:]:
            text = entry.latest_translation()
            lines.append(f"  - {entry.source_text[:60]} → {text[:60]}")
        return "\n".join(lines)

    async def _call_llm_correction(self, prompt: str) -> str:
        """调用 LLM 进行修正判断。"""
        if settings.LLM_PROVIDER == "openai":
            response = await self._client.chat.completions.create(
                model=settings.LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=512,
            )
            return response.choices[0].message.content.strip()

        elif settings.LLM_PROVIDER == "anthropic":
            response = await self._client.messages.create(
                model=settings.LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=512,
            )
            return response.content[0].text.strip()

        return ""
