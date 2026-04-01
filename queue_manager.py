"""
Task queue with abstract interface.

Replace InMemoryQueue with Redis/DB-backed implementation
by subclassing BaseQueue.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Optional

from schemas import TaskRecord, TaskStatus


class BaseQueue(ABC):
    """Abstract queue interface. Implement this to swap backends."""

    @abstractmethod
    async def put(self, task: TaskRecord) -> None:
        """Enqueue a task."""

    @abstractmethod
    async def get(self) -> TaskRecord:
        """Dequeue next task. Blocks until available."""

    @abstractmethod
    async def cancel(self, task_id: str) -> bool:
        """Cancel a pending task. Returns True if cancelled."""

    @abstractmethod
    async def get_task(self, task_id: str) -> Optional[TaskRecord]:
        """Get task by ID from any state."""

    @abstractmethod
    async def update_task(self, task: TaskRecord) -> None:
        """Update task record."""

    @abstractmethod
    async def list_tasks(
        self,
        status: Optional[TaskStatus] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[list[TaskRecord], int]:
        """List tasks with optional filtering. Returns (tasks, total_count)."""

    @abstractmethod
    async def pending_count(self) -> int:
        """Number of tasks waiting in queue."""

    @abstractmethod
    async def queue_position(self, task_id: str) -> Optional[int]:
        """Position in queue (1-based), or None if not pending."""


class InMemoryQueue(BaseQueue):
    """AsyncIO-based in-memory queue. Data lost on restart."""

    def __init__(self, maxsize: int = 100) -> None:
        self._queue: asyncio.Queue[str] = asyncio.Queue(maxsize=maxsize)
        self._tasks: OrderedDict[str, TaskRecord] = OrderedDict()
        self._pending_ids: list[str] = []  # ordered pending task ids

    async def put(self, task: TaskRecord) -> None:
        self._tasks[task.task_id] = task
        self._pending_ids.append(task.task_id)
        await self._queue.put(task.task_id)

    async def get(self) -> TaskRecord:
        while True:
            task_id = await self._queue.get()
            # Skip cancelled tasks
            task = self._tasks.get(task_id)
            if task and task.status == TaskStatus.PENDING:
                if task_id in self._pending_ids:
                    self._pending_ids.remove(task_id)
                return task

    async def cancel(self, task_id: str) -> bool:
        task = self._tasks.get(task_id)
        if task and task.status == TaskStatus.PENDING:
            task.status = TaskStatus.CANCELLED
            if task_id in self._pending_ids:
                self._pending_ids.remove(task_id)
            return True
        return False

    async def get_task(self, task_id: str) -> Optional[TaskRecord]:
        return self._tasks.get(task_id)

    async def update_task(self, task: TaskRecord) -> None:
        self._tasks[task.task_id] = task

    async def list_tasks(
        self,
        status: Optional[TaskStatus] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[list[TaskRecord], int]:
        tasks = list(self._tasks.values())
        if status:
            tasks = [t for t in tasks if t.status == status]
        # Newest first
        tasks.reverse()
        total = len(tasks)
        start = (page - 1) * page_size
        return tasks[start : start + page_size], total

    async def pending_count(self) -> int:
        return len(self._pending_ids)

    async def queue_position(self, task_id: str) -> Optional[int]:
        try:
            return self._pending_ids.index(task_id) + 1
        except ValueError:
            return None
