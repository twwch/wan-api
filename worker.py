"""
GPU worker that consumes tasks from the queue and runs Wan2.2 inference.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import random
import traceback
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Optional

from PIL import Image

from config import settings
from schemas import TaskRecord, TaskStatus, TaskType

logger = logging.getLogger(__name__)


def _decode_image(b64_str: str) -> Image.Image:
    """Decode base64 string to PIL Image."""
    data = base64.b64decode(b64_str)
    return Image.open(BytesIO(data)).convert("RGB")


def _patch_flash_attention():
    """Replace flash_attention with attention (has SDPA fallback) for non-flash_attn environments."""
    from wan.modules.attention import FLASH_ATTN_2_AVAILABLE, FLASH_ATTN_3_AVAILABLE
    if not FLASH_ATTN_2_AVAILABLE and not FLASH_ATTN_3_AVAILABLE:
        import wan.modules.attention as attn_mod
        import wan.modules.model as model_mod
        logger.info("flash_attn not available, patching to use PyTorch SDPA")
        attn_mod.flash_attention = attn_mod.attention
        model_mod.flash_attention = attn_mod.attention


def _load_model():
    """Load Wan2.2 TI2V-5B model. Called once at startup."""
    _patch_flash_attention()

    import wan
    from wan.configs import WAN_CONFIGS

    cfg = WAN_CONFIGS["ti2v-5B"]
    model = wan.WanTI2V(
        config=cfg,
        checkpoint_dir=settings.model_checkpoint_dir,
        device_id=settings.device_id,
        rank=0,
    )
    return model, cfg


SIZE_MAP = {
    "1280*704": (1280, 704),
    "704*1280": (704, 1280),
    "1024*704": (1024, 704),
    "704*1024": (704, 1024),
    "832*480": (832, 480),
    "480*832": (480, 832),
}


class Worker:
    def __init__(self) -> None:
        self._model: Any = None
        self._config: Any = None
        self._running = False
        self._ready = False
        self._start_error: Optional[str] = None
        self._current_task_id: Optional[str] = None
        self._notify_callbacks: dict[str, list[Callable]] = {}

    @property
    def current_task_id(self) -> Optional[str]:
        return self._current_task_id

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def start_error(self) -> Optional[str]:
        return self._start_error

    def register_callback(self, task_id: str, callback: Callable) -> None:
        self._notify_callbacks.setdefault(task_id, []).append(callback)

    def unregister_callback(self, task_id: str, callback: Callable) -> None:
        cbs = self._notify_callbacks.get(task_id, [])
        if callback in cbs:
            cbs.remove(callback)

    async def _notify(self, task: TaskRecord) -> None:
        for cb in self._notify_callbacks.get(task.task_id, []):
            try:
                await cb(task)
            except Exception:
                pass

    def load_model(self) -> None:
        """Load model in a thread to not block the event loop."""
        logger.info("Loading Wan2.2 TI2V-5B model...")
        self._model, self._config = _load_model()
        logger.info("Model loaded successfully.")

    async def start(self, queue) -> None:
        """Main worker loop. Runs as an asyncio task."""
        self._running = True
        # Load model in thread pool
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self.load_model)
        except Exception as e:
            self._start_error = f"{type(e).__name__}: {e}"
            logger.error(f"Worker failed to load model: {self._start_error}")
            return

        self._ready = True
        logger.info("Worker started, waiting for tasks...")
        while self._running:
            try:
                task = await queue.get()
                await self._process_task(task, queue)
            except asyncio.CancelledError:
                logger.info("Worker cancelled, shutting down...")
                break
            except Exception:
                logger.exception("Worker loop error")

    async def stop(self) -> None:
        self._running = False

    async def _process_task(self, task: TaskRecord, queue) -> None:
        self._current_task_id = task.task_id
        task.status = TaskStatus.PROCESSING
        task.started_at = datetime.now()
        await queue.update_task(task)
        await self._notify(task)

        try:
            video_path = await self._generate(task)
            task.status = TaskStatus.COMPLETED
            task.video_path = video_path
            task.progress = 1.0
            task.completed_at = datetime.now()
            logger.info(f"Task {task.task_id} completed: {video_path}")
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = f"{type(e).__name__}: {e}"
            task.completed_at = datetime.now()
            logger.error(f"Task {task.task_id} failed: {traceback.format_exc()}")
        finally:
            self._current_task_id = None
            await queue.update_task(task)
            await self._notify(task)

    async def _generate(self, task: TaskRecord) -> str:
        """Run Wan2.2 generation in thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._generate_sync, task)

    def _generate_sync(self, task: TaskRecord) -> str:
        """Synchronous generation on GPU."""
        from wan.utils.utils import save_video

        size_str = task.size or settings.default_size
        size = SIZE_MAP.get(size_str, (1280, 704))
        frame_num = task.frame_num or settings.default_frame_num
        sample_steps = task.sample_steps or settings.default_sample_steps
        shift = task.sample_shift or settings.default_sample_shift
        guide_scale = task.guide_scale or settings.default_guide_scale
        seed = task.seed if (task.seed is not None and task.seed >= 0) else random.randint(0, 2**32 - 1)

        # Determine image input
        img = None
        if task.task_type == TaskType.I2V and task.reference_image:
            img = _decode_image(task.reference_image)
        elif task.task_type == TaskType.TI2V and task.first_frame:
            img = _decode_image(task.first_frame)
        # t2v: img stays None

        # Build generation kwargs matching WanTI2V.generate() signature:
        #   generate(input_prompt, img=None, size=None, max_area=720*1280,
        #            frame_num=81, shift=5.0, sample_solver='unipc',
        #            sampling_steps=40, guide_scale=5.0, seed=-1,
        #            offload_model=True)
        w, h = size
        max_area = w * h

        kwargs = dict(
            img=img,
            size=size,
            max_area=max_area,
            frame_num=frame_num,
            shift=shift,
            sample_solver='unipc',
            sampling_steps=sample_steps,
            guide_scale=guide_scale,
            seed=seed,
            offload_model=settings.offload_model,
        )

        # Last frame support (if model accepts it)
        if task.last_frame:
            last_img = _decode_image(task.last_frame)
            kwargs["last_img"] = last_img

        video = self._model.generate(task.prompt, **kwargs)

        # Save video
        output_path = str(Path(settings.output_dir) / f"{task.task_id}.mp4")
        save_video(
            video[None],
            output_path,
            fps=self._config.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1),
        )
        return output_path
