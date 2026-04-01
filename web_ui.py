"""
Gradio UI for Wan2.2 Video Generation, mounted at /app.

Directly interacts with the queue/worker instead of HTTP self-requests,
avoiding 502 issues on Windows and reducing latency.
"""

from __future__ import annotations

import asyncio
import base64
from io import BytesIO
from typing import TYPE_CHECKING

import logging

import gradio as gr

if TYPE_CHECKING:
    from queue_manager import BaseQueue
    from worker import Worker

logger = logging.getLogger(__name__)

# Set by mount_to_app()
_queue: BaseQueue | None = None
_worker: Worker | None = None


def _image_to_base64(img) -> str | None:
    if img is None:
        return None
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


async def submit_task(
    task_type: str,
    prompt: str,
    reference_image,
    first_frame,
    last_frame,
    size: str,
    frame_num: int,
    sample_steps: int,
    guide_scale: float,
    seed: int,
):
    from schemas import TaskRecord

    if _worker and _worker.start_error:
        raise gr.Error(f"Worker 启动失败: {_worker.start_error}")

    if task_type == "i2v" and reference_image is None:
        raise gr.Error("i2v 模式需要上传参考图")
    if task_type == "ti2v" and first_frame is None:
        raise gr.Error("ti2v 模式需要上传首帧图")

    ref_img_b64 = _image_to_base64(reference_image) if task_type == "i2v" else None
    first_b64 = _image_to_base64(first_frame) if task_type == "ti2v" else None
    last_b64 = _image_to_base64(last_frame) if task_type == "ti2v" else None

    task = TaskRecord(
        task_type=task_type,
        prompt=prompt,
        reference_image=ref_img_b64,
        first_frame=first_b64,
        last_frame=last_b64,
        size=size,
        frame_num=int(frame_num),
        sample_steps=int(sample_steps),
        guide_scale=float(guide_scale),
        seed=int(seed),
    )

    try:
        await _queue.put(task)
    except asyncio.QueueFull:
        raise gr.Error("队列已满，请稍后再试")

    logger.info(f"Task {task.task_id} created ({task.task_type})")
    position = await _queue.queue_position(task.task_id)
    data = _task_to_dict(task, position)
    return (
        f"任务已提交: {task.task_id}",
        task.task_id,
        _format_status(data),
        gr.update(interactive=True),
        None,
    )


async def poll_status(task_id: str):
    if not task_id:
        return "请先提交任务", gr.update(), None

    task = await _queue.get_task(task_id)
    if not task:
        return "任务不存在", gr.update(), None

    position = await _queue.queue_position(task_id)
    data = _task_to_dict(task, position)
    status_text = _format_status(data)

    video = None
    if task.status == "completed" and task.video_path:
        video = task.video_path

    return status_text, gr.update(), video


async def auto_poll(task_id: str):
    """Poll until terminal state."""
    if not task_id:
        yield "请先提交任务", None
        return

    while True:
        task = await _queue.get_task(task_id)
        if not task:
            yield "任务不存在", None
            return

        position = await _queue.queue_position(task_id)
        data = _task_to_dict(task, position)
        status_text = _format_status(data)

        if task.status == "completed" and task.video_path:
            yield status_text, task.video_path
            return

        if task.status in ("failed", "cancelled"):
            yield status_text, None
            return

        if _worker and not _worker.ready and not _worker.start_error:
            status_text += "\n\n⏳ 模型加载中，请等待..."
        elif _worker and _worker.start_error:
            status_text += f"\n\n❌ Worker 错误: {_worker.start_error}"
            yield status_text, None
            return
        yield status_text, None
        await asyncio.sleep(3)


async def get_queue_status():
    try:
        all_tasks = list((await _queue.list_tasks(page=1, page_size=10000))[0])
        from schemas import TaskStatus
        completed = sum(1 for t in all_tasks if t.status == TaskStatus.COMPLETED)
        failed = sum(1 for t in all_tasks if t.status == TaskStatus.FAILED)
        pending = await _queue.pending_count()
        current_id = _worker.current_task_id if _worker else None
        lines = [
            f"排队任务: {pending}",
            f"当前任务: {current_id or '无'}",
            f"已完成: {completed}",
            f"已失败: {failed}",
        ]
        return "\n".join(lines)
    except Exception as e:
        return f"获取失败: {e}"


async def cancel_task(task_id: str):
    if not task_id:
        return "无任务可取消"
    task = await _queue.get_task(task_id)
    if not task:
        return "任务不存在"
    if task.status != "pending":
        return f"无法取消 {task.status} 状态的任务"
    ok = await _queue.cancel(task_id)
    if ok:
        return "任务已取消"
    return "取消失败"


def _task_to_dict(task, position=None) -> dict:
    """Convert TaskRecord to a dict for display."""
    return {
        "task_id": task.task_id,
        "task_type": task.task_type,
        "status": task.status,
        "prompt": task.prompt,
        "progress": task.progress,
        "error": task.error,
        "video_path": task.video_path,
        "queue_position": position,
    }


def _format_status(data: dict) -> str:
    status_emoji = {
        "pending": "🕐 排队中",
        "processing": "⚙️ 生成中",
        "completed": "✅ 完成",
        "failed": "❌ 失败",
        "cancelled": "🚫 已取消",
    }
    lines = [
        f"状态: {status_emoji.get(data['status'], data['status'])}",
        f"任务ID: {data['task_id']}",
        f"类型: {data['task_type']}",
        f"提示词: {data['prompt'][:80]}",
    ]
    if data.get("queue_position"):
        lines.append(f"队列位置: 第 {data['queue_position']} 位")
    if data.get("progress"):
        lines.append(f"进度: {data['progress'] * 100:.0f}%")
    if data.get("error"):
        lines.append(f"错误: {data['error']}")
    return "\n".join(lines)


def on_task_type_change(task_type: str):
    """Show/hide image inputs based on task type."""
    return (
        gr.update(visible=task_type == "i2v"),  # reference_image
        gr.update(visible=task_type == "ti2v"),  # first_frame
        gr.update(visible=task_type == "ti2v"),  # last_frame
    )


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Wan2.2 视频生成", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Wan2.2 视频生成平台")

        with gr.Row():
            # ─── Left: Inputs ───
            with gr.Column(scale=1):
                task_type = gr.Radio(
                    choices=["t2v", "i2v", "ti2v"],
                    value="t2v",
                    label="任务类型",
                    info="t2v: 文生视频 | i2v: 参考图生视频 | ti2v: 首尾帧生视频",
                )
                prompt = gr.Textbox(
                    label="提示词",
                    placeholder="描述你想生成的视频...",
                    lines=3,
                )

                reference_image = gr.Image(
                    label="参考图 (i2v)",
                    type="pil",
                    visible=False,
                )
                first_frame = gr.Image(
                    label="首帧图 (ti2v)",
                    type="pil",
                    visible=False,
                )
                last_frame = gr.Image(
                    label="尾帧图 (ti2v, 可选)",
                    type="pil",
                    visible=False,
                )

                with gr.Accordion("高级参数", open=False):
                    size = gr.Dropdown(
                        choices=["1280*704", "704*1280", "1024*704", "704*1024", "832*480", "480*832"],
                        value="1280*704",
                        label="分辨率",
                    )
                    frame_num = gr.Slider(
                        minimum=17, maximum=241, value=121, step=4,
                        label="帧数 (4n+1)",
                    )
                    sample_steps = gr.Slider(
                        minimum=10, maximum=100, value=50, step=1,
                        label="采样步数",
                    )
                    guide_scale = gr.Slider(
                        minimum=1.0, maximum=15.0, value=5.0, step=0.5,
                        label="引导强度",
                    )
                    seed = gr.Number(value=-1, label="种子 (-1=随机)", precision=0)

                with gr.Row():
                    submit_btn = gr.Button("提交任务", variant="primary")
                    cancel_btn = gr.Button("取消任务", variant="stop")

            # ─── Right: Output ───
            with gr.Column(scale=1):
                status_text = gr.Textbox(label="任务状态", lines=6, interactive=False)
                video_output = gr.Video(label="生成结果")
                task_id_state = gr.State("")

                with gr.Row():
                    refresh_btn = gr.Button("刷新状态")
                    queue_btn = gr.Button("查看队列")
                queue_text = gr.Textbox(label="队列状态", lines=4, interactive=False)

        # ─── Events ───

        task_type.change(
            on_task_type_change,
            inputs=[task_type],
            outputs=[reference_image, first_frame, last_frame],
        )

        submit_btn.click(
            submit_task,
            inputs=[task_type, prompt, reference_image, first_frame, last_frame,
                    size, frame_num, sample_steps, guide_scale, seed],
            outputs=[status_text, task_id_state, status_text, refresh_btn, video_output],
        ).then(
            auto_poll,
            inputs=[task_id_state],
            outputs=[status_text, video_output],
        )

        refresh_btn.click(
            poll_status,
            inputs=[task_id_state],
            outputs=[status_text, refresh_btn, video_output],
        )

        cancel_btn.click(
            cancel_task,
            inputs=[task_id_state],
            outputs=[status_text],
        )

        queue_btn.click(get_queue_status, outputs=[queue_text])

    return demo


def mount_to_app(app, queue, worker):
    """Mount Gradio UI to FastAPI app at /app."""
    global _queue, _worker
    _queue = queue
    _worker = worker
    demo = build_ui()
    return gr.mount_gradio_app(app, demo, path="/app")
