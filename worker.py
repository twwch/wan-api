"""
GPU worker that consumes tasks from the queue and runs Wan2.2 inference
using HuggingFace Diffusers pipeline.
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

SAMPLE_FPS = 24

SIZE_MAP = {
    "1280*704": (1280, 704),
    "704*1280": (704, 1280),
    "1024*704": (1024, 704),
    "704*1024": (704, 1024),
    "832*480": (832, 480),
    "480*832": (480, 832),
}


def _decode_image(b64_str: str) -> Image.Image:
    """Decode base64 string to PIL Image."""
    data = base64.b64decode(b64_str)
    return Image.open(BytesIO(data)).convert("RGB")


def _load_pipeline():
    """Load Wan2.2 TI2V-5B Diffusers pipeline."""
    import torch
    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline, WanPipeline

    model_id = settings.model_checkpoint_dir
    dtype = torch.float16

    logger.info(f"Loading VAE from {model_id} ...")
    vae = AutoencoderKLWan.from_pretrained(
        model_id, subfolder="vae", torch_dtype=torch.float32
    )

    # T2V pipeline (text-only)
    logger.info(f"Loading T2V pipeline from {model_id} ...")
    t2v_pipe = WanPipeline.from_pretrained(
        model_id, vae=vae, torch_dtype=dtype
    )

    # I2V pipeline (image+text) — shares components with t2v
    logger.info(f"Loading I2V pipeline from {model_id} ...")
    i2v_pipe = WanImageToVideoPipeline.from_pretrained(
        model_id, vae=vae, torch_dtype=dtype
    )

    # Enable VAE tiling to avoid OOM during decode
    t2v_pipe.vae.enable_tiling()
    i2v_pipe.vae.enable_tiling()

    if settings.offload_model:
        t2v_pipe.enable_model_cpu_offload()
        i2v_pipe.enable_model_cpu_offload()
    else:
        t2v_pipe.to(f"cuda:{settings.device_id}")
        i2v_pipe.to(f"cuda:{settings.device_id}")

    return t2v_pipe, i2v_pipe


class Worker:
    def __init__(self) -> None:
        self._t2v_pipe: Any = None
        self._i2v_pipe: Any = None
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
        logger.info("Loading Wan2.2 TI2V-5B Diffusers pipeline...")
        self._t2v_pipe, self._i2v_pipe = _load_pipeline()
        logger.info("Pipeline loaded successfully.")

    async def start(self, queue) -> None:
        """Main worker loop. Runs as an asyncio task."""
        self._running = True
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
        """Run generation in thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._generate_sync, task)

    def _generate_sync(self, task: TaskRecord) -> str:
        """Synchronous generation on GPU using Diffusers pipeline."""
        import torch
        from diffusers.utils import export_to_video

        size_str = task.size or settings.default_size
        w, h = SIZE_MAP.get(size_str, (1280, 704))
        num_frames = task.frame_num or settings.default_frame_num
        num_steps = task.sample_steps or settings.default_sample_steps
        guide_scale = task.guide_scale or settings.default_guide_scale
        seed = task.seed if (task.seed is not None and task.seed >= 0) else random.randint(0, 2**32 - 1)

        generator = torch.Generator(device="cpu").manual_seed(seed)

        # Determine image input
        img = None
        if task.task_type == TaskType.I2V and task.reference_image:
            img = _decode_image(task.reference_image)
        elif task.task_type == TaskType.TI2V and task.first_frame:
            img = _decode_image(task.first_frame)

        # Choose pipeline based on whether we have an image
        if img is not None:
            # Image-to-video pipeline
            output = self._i2v_pipe(
                prompt=task.prompt,
                image=img,
                height=h,
                width=w,
                num_frames=num_frames,
                num_inference_steps=num_steps,
                guidance_scale=guide_scale,
                generator=generator,
            )
        else:
            # Text-to-video pipeline
            output = self._t2v_pipe(
                prompt=task.prompt,
                height=h,
                width=w,
                num_frames=num_frames,
                num_inference_steps=num_steps,
                guidance_scale=guide_scale,
                generator=generator,
            )

        # Save video
        frames = output.frames[0]
        output_path = str(Path(settings.output_dir) / f"{task.task_id}.mp4")
        export_to_video(frames, output_path, fps=SAMPLE_FPS)
        return output_path
