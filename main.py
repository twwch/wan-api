"""
Wan2.2 Video Generation API Service
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from config import settings
from queue_manager import InMemoryQueue
from schemas import (
    QueueStatus,
    TaskCreate,
    TaskInfo,
    TaskListResponse,
    TaskRecord,
    TaskStatus,
    TaskType,
)
from worker import Worker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

queue = InMemoryQueue(maxsize=settings.max_queue_size)
worker = Worker()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: launch worker
    worker_task = asyncio.create_task(worker.start(queue))
    logger.info("Wan2.2 API service started")
    yield
    # Shutdown: stop worker gracefully
    await worker.stop()
    worker_task.cancel()
    try:
        await worker_task
    except asyncio.CancelledError:
        pass
    logger.info("Wan2.2 API service stopped")


app = FastAPI(
    title="Wan2.2 Video Generation API",
    version="1.0.0",
    lifespan=lifespan,
)

# Mount Gradio UI at /app
from web_ui import mount_to_app

mount_to_app(app)


def _validate_task(req: TaskCreate) -> None:
    """Validate task inputs match task type."""
    if req.task_type == TaskType.I2V and not req.reference_image:
        raise HTTPException(400, "reference_image is required for i2v task")
    if req.task_type == TaskType.TI2V and not req.first_frame:
        raise HTTPException(400, "first_frame is required for ti2v task")


def _task_to_info(task: TaskRecord, position: Optional[int] = None) -> TaskInfo:
    video_url = f"/api/v1/tasks/{task.task_id}/video" if task.video_path else None
    return TaskInfo(
        task_id=task.task_id,
        task_type=task.task_type,
        status=task.status,
        prompt=task.prompt,
        created_at=task.created_at,
        started_at=task.started_at,
        completed_at=task.completed_at,
        progress=task.progress,
        error=task.error,
        video_url=video_url,
        queue_position=position,
    )


# ─── Task Endpoints ───


@app.post("/api/v1/tasks", response_model=TaskInfo, status_code=201)
async def create_task(req: TaskCreate):
    """Submit a video generation task."""
    _validate_task(req)

    task = TaskRecord(
        task_type=req.task_type,
        prompt=req.prompt,
        reference_image=req.reference_image,
        first_frame=req.first_frame,
        last_frame=req.last_frame,
        size=req.size,
        frame_num=req.frame_num,
        sample_steps=req.sample_steps,
        sample_shift=req.sample_shift,
        guide_scale=req.guide_scale,
        seed=req.seed,
    )

    try:
        await queue.put(task)
    except asyncio.QueueFull:
        raise HTTPException(503, "Queue is full, try again later")

    position = await queue.queue_position(task.task_id)
    logger.info(f"Task {task.task_id} created ({task.task_type}), position: {position}")
    return _task_to_info(task, position)


@app.get("/api/v1/tasks/{task_id}", response_model=TaskInfo)
async def get_task(task_id: str):
    """Get task status and details."""
    task = await queue.get_task(task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    position = await queue.queue_position(task_id)
    return _task_to_info(task, position)


@app.get("/api/v1/tasks/{task_id}/video")
async def get_video(task_id: str):
    """Download generated video."""
    task = await queue.get_task(task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    if task.status != TaskStatus.COMPLETED or not task.video_path:
        raise HTTPException(400, f"Video not ready, task status: {task.status}")
    return FileResponse(
        task.video_path,
        media_type="video/mp4",
        filename=f"{task_id}.mp4",
    )


@app.delete("/api/v1/tasks/{task_id}")
async def cancel_task(task_id: str):
    """Cancel a pending task."""
    task = await queue.get_task(task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    if task.status != TaskStatus.PENDING:
        raise HTTPException(400, f"Cannot cancel task in {task.status} status")
    ok = await queue.cancel(task_id)
    if not ok:
        raise HTTPException(400, "Failed to cancel task")
    return {"message": "Task cancelled", "task_id": task_id}


@app.get("/api/v1/tasks", response_model=TaskListResponse)
async def list_tasks(
    status: Optional[TaskStatus] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    """List tasks with optional status filter."""
    tasks, total = await queue.list_tasks(status=status, page=page, page_size=page_size)
    infos = []
    for t in tasks:
        pos = await queue.queue_position(t.task_id)
        infos.append(_task_to_info(t, pos))
    return TaskListResponse(tasks=infos, total=total, page=page, page_size=page_size)


# ─── Queue Status ───


@app.get("/api/v1/queue/status", response_model=QueueStatus)
async def get_queue_status():
    """Get current queue status."""
    all_tasks = list((await queue.list_tasks(page=1, page_size=10000))[0])
    completed = sum(1 for t in all_tasks if t.status == TaskStatus.COMPLETED)
    failed = sum(1 for t in all_tasks if t.status == TaskStatus.FAILED)
    pending = await queue.pending_count()

    current_id = worker.current_task_id
    current_progress = 0.0
    if current_id:
        current_task = await queue.get_task(current_id)
        if current_task:
            current_progress = current_task.progress

    return QueueStatus(
        queue_size=pending,
        current_task_id=current_id,
        current_task_progress=current_progress,
        pending_count=pending,
        completed_count=completed,
        failed_count=failed,
    )


# ─── WebSocket ───


@app.websocket("/ws/tasks/{task_id}")
async def task_ws(websocket: WebSocket, task_id: str):
    """WebSocket for real-time task status updates."""
    task = await queue.get_task(task_id)
    if not task:
        await websocket.close(code=4004, reason="Task not found")
        return

    await websocket.accept()

    update_queue: asyncio.Queue = asyncio.Queue()

    async def on_update(updated_task: TaskRecord):
        await update_queue.put(updated_task)

    worker.register_callback(task_id, on_update)

    try:
        # Send current state
        pos = await queue.queue_position(task_id)
        info = _task_to_info(task, pos)
        await websocket.send_text(info.model_dump_json())

        # If already terminal, close
        if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
            return

        # Wait for updates
        while True:
            updated = await update_queue.get()
            pos = await queue.queue_position(task_id)
            info = _task_to_info(updated, pos)
            await websocket.send_text(info.model_dump_json())
            if updated.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
                break
    except WebSocketDisconnect:
        pass
    finally:
        worker.unregister_callback(task_id, on_update)


# ─── Health ───


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=settings.host, port=settings.port, reload=False)
