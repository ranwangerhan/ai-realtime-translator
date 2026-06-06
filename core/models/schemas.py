"""
数据模型 / 消息协议

定义了流水线中各个阶段之间传递的结构化事件。
所有模型均为 pydantic BaseModel，天然支持序列化与校验。
"""

import time
from typing import Literal, Optional, List

from pydantic import BaseModel, Field


# ═══════════════════════════════════════════
# ASR 阶段
# ═══════════════════════════════════════════

class ASREvent(BaseModel):
    """ASR 引擎输出的单句识别结果。

    当 VAD 检测到一句话结束后，ASR 对这段音频进行识别，
    产生一个 ASREvent 送入翻译管道。
    """

    seq_id: int                         # 句子序号，单调递增
    text: str                           # 识别文本
    start_time: float                   # 音频起始时间戳 (time.monotonic)
    end_time: float                     # 音频结束时间戳
    duration_ms: float = Field(0.0, ge=0)  # 音频时长 (ms)，可由 end-start 算得
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="ASR 置信度")
    is_final: bool = True               # True=定稿; False=中间结果（流式预览用）

    def __str__(self) -> str:
        return f"[{self.seq_id:04d}] c={self.confidence:.2f} \"{self.text[:60]}\""


# ═══════════════════════════════════════════
# LLM 翻译阶段
# ═══════════════════════════════════════════

class TransEvent(BaseModel):
    """LLM 翻译结果，携带修正链信息。"""

    seq_id: int
    source_text: str                     # ASR 原文
    translated_text: str                 # 中文译文
    status: Literal["streaming", "final", "corrected"] = "final"
    correction_of: Optional[int] = None  # 若为修正事件，指向原始句子的 seq_id
    correction_reason: str = ""          # 触发修正的原因
    timestamp: float = Field(default_factory=time.time)

    @property
    def is_correction(self) -> bool:
        return self.correction_of is not None

    def __str__(self) -> str:
        tag = "修正" if self.is_correction else self.status
        return f"[{self.seq_id:04d}][{tag}] \"{self.translated_text[:60]}\""


# ═══════════════════════════════════════════
# 滑动窗口修正阶段
# ═══════════════════════════════════════════

class CorrectionEvent(BaseModel):
    """下发给前端的修正指令。"""

    seq_id: int
    original_translation: str            # 修正前的译文
    corrected_translation: str           # 修正后的译文
    original_source: Optional[str] = None   # 修正前的 ASR 原文
    corrected_source: Optional[str] = None  # 修正后的 ASR 原文（如 ASR 也重跑了）
    reason: str = ""                     # 修正理由（供前端展示）
    timestamp: float = Field(default_factory=time.time)


class CorrectionTask(BaseModel):
    """滑动窗口内部下发给 LLM 的重审工单。"""

    seq_id: int
    original_text: str
    context_before: List[str] = Field(default_factory=list)  # 前文窗口
    context_after: List[str] = Field(default_factory=list)   # 后文窗口
    reason: str = ""
    created_at: float = Field(default_factory=time.time)


# ═══════════════════════════════════════════
# 窗口条目（SWC 内部状态）
# ═══════════════════════════════════════════

class SlidingWindowEntry(BaseModel):
    """滑动窗口中的单句完整状态。"""

    seq_id: int
    source_text: str
    translated_text: str
    asr_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    is_corrected: bool = False
    correction_events: List[CorrectionEvent] = Field(default_factory=list)
    arrived_at: float = Field(default_factory=time.time)

    def latest_translation(self) -> str:
        """返回最新（含修正后）的译文。"""
        if self.correction_events:
            return self.correction_events[-1].corrected_translation
        return self.translated_text

    def __str__(self) -> str:
        return f"[{self.seq_id:04d}] \"{self.source_text[:40]}\" → \"{self.latest_translation()[:40]}\""
