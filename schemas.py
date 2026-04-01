from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class TaskType(str, Enum):
    T2V = "t2v"  # text-to-video
    I2V = "i2v"  # reference image-to-video
    TI2V = "ti2v"  # first frame (+last frame) to video


class TaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskCreate(BaseModel):
    task_type: TaskType
    prompt: str = Field(..., min_length=1, max_length=2048)

    # Image inputs (base64 encoded)
    reference_image: Optional[str] = None  # for i2v
    first_frame: Optional[str] = None  # for ti2v
    last_frame: Optional[str] = None  # for ti2v (optional)

    # Generation parameters (None = use defaults)
    size: Optional[str] = None  # e.g. "1280*704"
    frame_num: Optional[int] = Field(None, ge=5, le=241)
    sample_steps: Optional[int] = Field(None, ge=1, le=100)
    sample_shift: Optional[float] = Field(None, ge=0.0, le=50.0)
    guide_scale: Optional[float] = Field(None, ge=0.0, le=20.0)
    seed: Optional[int] = None


class TaskInfo(BaseModel):
    task_id: str
    task_type: TaskType
    status: TaskStatus
    prompt: str
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = 0.0  # 0.0 ~ 1.0
    error: Optional[str] = None
    video_url: Optional[str] = None
    queue_position: Optional[int] = None


class TaskRecord(BaseModel):
    """Internal full task record."""

    task_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    task_type: TaskType
    status: TaskStatus = TaskStatus.PENDING
    prompt: str

    reference_image: Optional[str] = None
    first_frame: Optional[str] = None
    last_frame: Optional[str] = None

    size: Optional[str] = None
    frame_num: Optional[int] = None
    sample_steps: Optional[int] = None
    sample_shift: Optional[float] = None
    guide_scale: Optional[float] = None
    seed: Optional[int] = None

    created_at: datetime = Field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = 0.0
    error: Optional[str] = None
    video_path: Optional[str] = None


class QueueStatus(BaseModel):
    queue_size: int
    current_task_id: Optional[str] = None
    current_task_progress: float = 0.0
    pending_count: int
    completed_count: int
    failed_count: int


class TaskListResponse(BaseModel):
    tasks: list[TaskInfo]
    total: int
    page: int
    page_size: int
