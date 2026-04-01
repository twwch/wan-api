"""
Gradio UI for Wan2.2 Video Generation, mounted at /app.
"""

from __future__ import annotations

import base64
import json
import time
from io import BytesIO

import gradio as gr
import httpx

API_BASE = "http://127.0.0.1:8000"


def _image_to_base64(img) -> str | None:
    if img is None:
        return None
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def submit_task(
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
    payload = {
        "task_type": task_type,
        "prompt": prompt,
        "size": size,
        "frame_num": int(frame_num),
        "sample_steps": int(sample_steps),
        "guide_scale": float(guide_scale),
        "seed": int(seed),
    }

    if task_type == "i2v":
        if reference_image is None:
            raise gr.Error("i2v 模式需要上传参考图")
        payload["reference_image"] = _image_to_base64(reference_image)
    elif task_type == "ti2v":
        if first_frame is None:
            raise gr.Error("ti2v 模式需要上传首帧图")
        payload["first_frame"] = _image_to_base64(first_frame)
        if last_frame is not None:
            payload["last_frame"] = _image_to_base64(last_frame)

    resp = httpx.post(f"{API_BASE}/api/v1/tasks", json=payload, timeout=10)
    if resp.status_code != 201:
        raise gr.Error(f"提交失败: {resp.text}")

    data = resp.json()
    task_id = data["task_id"]
    return (
        f"任务已提交: {task_id}",
        task_id,
        _format_status(data),
        gr.update(interactive=True),
        None,
    )


def poll_status(task_id: str):
    if not task_id:
        return "请先提交任务", gr.update(), None

    resp = httpx.get(f"{API_BASE}/api/v1/tasks/{task_id}", timeout=10)
    if resp.status_code != 200:
        return f"查询失败: {resp.text}", gr.update(), None

    data = resp.json()
    status_text = _format_status(data)

    video = None
    if data["status"] == "completed" and data.get("video_url"):
        video_resp = httpx.get(f"{API_BASE}{data['video_url']}", timeout=60)
        if video_resp.status_code == 200:
            tmp_path = f"/tmp/wan_{task_id}.mp4"
            with open(tmp_path, "wb") as f:
                f.write(video_resp.content)
            video = tmp_path

    return status_text, gr.update(), video


def auto_poll(task_id: str):
    """Poll until terminal state."""
    if not task_id:
        yield "请先提交任务", None
        return

    while True:
        resp = httpx.get(f"{API_BASE}/api/v1/tasks/{task_id}", timeout=10)
        if resp.status_code != 200:
            yield f"查询失败: {resp.text}", None
            return

        data = resp.json()
        status = data["status"]
        status_text = _format_status(data)

        if status == "completed" and data.get("video_url"):
            video_resp = httpx.get(f"{API_BASE}{data['video_url']}", timeout=120)
            if video_resp.status_code == 200:
                tmp_path = f"/tmp/wan_{task_id}.mp4"
                with open(tmp_path, "wb") as f:
                    f.write(video_resp.content)
                yield status_text, tmp_path
                return

        if status in ("failed", "cancelled"):
            yield status_text, None
            return

        yield status_text, None
        time.sleep(3)


def get_queue_status():
    try:
        resp = httpx.get(f"{API_BASE}/api/v1/queue/status", timeout=5)
        data = resp.json()
        lines = [
            f"排队任务: {data['pending_count']}",
            f"当前任务: {data['current_task_id'] or '无'}",
            f"已完成: {data['completed_count']}",
            f"已失败: {data['failed_count']}",
        ]
        return "\n".join(lines)
    except Exception as e:
        return f"获取失败: {e}"


def cancel_task(task_id: str):
    if not task_id:
        return "无任务可取消"
    resp = httpx.delete(f"{API_BASE}/api/v1/tasks/{task_id}", timeout=10)
    if resp.status_code == 200:
        return "任务已取消"
    return f"取消失败: {resp.text}"


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


def mount_to_app(app):
    """Mount Gradio UI to FastAPI app at /app."""
    demo = build_ui()
    return gr.mount_gradio_app(app, demo, path="/app")
