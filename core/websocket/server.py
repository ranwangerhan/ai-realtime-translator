"""
WebSocket 服务模块

基于 FastAPI + WebSocket 提供实时结果推送。
翻译结果和修正事件以 JSON 格式推送到前端。

消息协议:

服务端 → 客户端:
  {"type": "translation", "data": {TransEvent JSON}}
  {"type": "correction",  "data": {CorrectionEvent JSON}}
  {"type": "status",      "data": {"status": "connected"}}

客户端 → 服务端:
  {"type": "ping"}
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from typing import Any, Optional

from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from loguru import logger

from config import settings
from core.models.schemas import TransEvent, CorrectionEvent


class ConnectionManager:
    """WebSocket 连接管理器。"""

    def __init__(self) -> None:
        self._connections: dict[WebSocket, str] = {}  # ws → client_id

    @property
    def count(self) -> int:
        return len(self._connections)

    async def connect(self, ws: WebSocket) -> str:
        await ws.accept()
        client_id = f"client-{id(ws):08x}"
        self._connections[ws] = client_id
        logger.info("WebSocket 客户端已连接 | {} (当前 {} 个连接)", client_id, self.count)
        return client_id

    def disconnect(self, ws: WebSocket) -> None:
        client_id = self._connections.pop(ws, "unknown")
        logger.info("WebSocket 客户端已断开 | {} (剩余 {} 个连接)", client_id, self.count)

    async def broadcast(self, message: dict[str, Any]) -> None:
        """向所有连接的客户端广播消息。"""
        payload = json.dumps(message, ensure_ascii=False)
        dead: list[WebSocket] = []

        for ws in self._connections:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)

        for ws in dead:
            self.disconnect(ws)

    async def broadcast_translation(self, event: TransEvent) -> None:
        """广播翻译结果。"""
        await self.broadcast({
            "type": "translation",
            "data": event.model_dump(),
        })

    async def broadcast_asr(self, text: str, seq_id: int, confidence: float) -> None:
        """广播 ASR 原始文本（先于翻译显示）。"""
        await self.broadcast({
            "type": "asr_interim",
            "data": {
                "seq_id": seq_id,
                "text": text,
                "confidence": confidence,
                "timestamp": __import__("time").time(),
            },
        })

    async def broadcast_correction(self, event: CorrectionEvent) -> None:
        """广播修正事件。"""
        logger.info("广播修正事件: seq={} \"{}\" → \"{}\"",
                     event.seq_id, event.original_translation[:30], event.corrected_translation[:30])
        await self.broadcast({
            "type": "correction",
            "data": event.model_dump(),
        })

    async def broadcast_correction_status(self, status: str, checked: int = 0) -> None:
        """广播修正引擎状态。"""
        await self.broadcast({
            "type": "correction_status",
            "data": {"status": status, "checked": checked},
        })

    async def broadcast_status(self, status: str, detail: str = "") -> None:
        """广播状态消息。"""
        await self.broadcast({
            "type": "status",
            "data": {"status": status, "detail": detail},
        })


# 全局单例
manager = ConnectionManager()


# ──── FastAPI 应用 ──────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理。"""
    logger.info("WebSocket 服务启动 | ws://{}:{}", settings.WS_HOST, settings.WS_PORT)
    yield
    logger.info("WebSocket 服务停止")


app = FastAPI(lifespan=lifespan)

# ──── 前端页面 ────────────────────────────────

_FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend"
_INDEX_HTML = _FRONTEND_DIR / "index.html"


@app.get("/")
async def index():
    """提供前端页面。"""
    if _INDEX_HTML.exists():
        content = _INDEX_HTML.read_text(encoding="utf-8")
        return HTMLResponse(content)
    return HTMLResponse("<h1>AI 同声传译助手</h1><p>前端页面未找到</p>")


@app.get("/health")
async def health():
    return {"status": "ok", "connections": manager.count}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """WebSocket 端点。"""
    client_id = await manager.connect(ws)

    try:
        while True:
            data = await ws.receive_text()
            try:
                msg = json.loads(data)
                if msg.get("type") == "ping":
                    await ws.send_text(json.dumps({"type": "pong"}))
            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(ws)


# ──── 推送任务 ──────────────────────────────────────

class WebSocketServer:
    """异步推送任务：从队列读取事件并广播到 WebSocket。

    同时监听两个队列：
    - input_queue: TransEvent / CorrectionEvent（翻译+修正结果）
    - asr_queue:   ASREvent（原始 ASR 文本，先于翻译显示）
    """

    def __init__(
        self,
        input_queue: asyncio.Queue[TransEvent | CorrectionEvent],
        asr_queue: asyncio.Queue | None = None,
        correction_status_queue: asyncio.Queue | None = None,
    ) -> None:
        self._input_queue = input_queue
        self._asr_queue = asr_queue
        self._correction_status_queue = correction_status_queue
        self._running: bool = False

    async def run(self) -> None:
        """推送主循环。"""
        self._running = True
        logger.info("WebSocket 推送任务已启动")

        try:
            while self._running:
                # 同时等待三个队列，谁先来就处理谁
                tasks = [asyncio.create_task(self._input_queue.get())]
                if self._asr_queue is not None:
                    tasks.append(asyncio.create_task(self._asr_queue.get()))
                if self._correction_status_queue is not None:
                    tasks.append(asyncio.create_task(self._correction_status_queue.get()))

                done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for t in tasks:
                    if not t.done():
                        t.cancel()

                for t in done:
                    event = t.result()
                    try:
                        from core.models.schemas import ASREvent
                        if isinstance(event, TransEvent):
                            await manager.broadcast_translation(event)
                        elif isinstance(event, CorrectionEvent):
                            await manager.broadcast_correction(event)
                        elif isinstance(event, ASREvent):
                            await manager.broadcast_asr(
                                text=event.text,
                                seq_id=event.seq_id,
                                confidence=event.confidence,
                            )
                        elif isinstance(event, dict) and event.get("type") == "correction_status":
                            await manager.broadcast_correction_status(**event["data"])
                    except Exception:
                        pass
        except asyncio.CancelledError:
            logger.debug("WebSocket 推送任务被取消")
            self._running = False
            raise

    def stop(self) -> None:
        self._running = False


def start_uvicorn():
    """启动 uvicorn 服务器（在独立线程中运行）。"""
    import uvicorn
    uvicorn.run(
        app,
        host=settings.WS_HOST,
        port=settings.WS_PORT,
        log_level="info",
    )
