# Wan2.2 Video Generation API

基于 [Wan2.2](https://github.com/Wan-Video/Wan2.2) TI2V-5B 模型的视频生成服务，FastAPI + Gradio 前端，内置任务排队机制。

## 功能

- **文生视频 (t2v)** — 纯文本 prompt 生成视频
- **参考图生视频 (i2v)** — 参考图 + prompt 生成视频
- **首尾帧生视频 (ti2v)** — 首帧图（+ 可选尾帧）+ prompt 生成视频
- **任务排队** — 异步队列，单 GPU 串行处理，支持取消
- **实时状态** — WebSocket 推送 + 轮询
- **Gradio UI** — 可视化操作界面

## 环境要求

- Python 3.10+
- NVIDIA GPU ≥ 24GB VRAM（推荐 RTX 4090 / 5090）
- [Wan2.2](https://github.com/Wan-Video/Wan2.2) 及其依赖已安装
- TI2V-5B 模型权重已下载

## 安装

```bash
# 1. 安装 Wan2.2（参考官方文档）
git clone https://github.com/Wan-Video/Wan2.2.git
cd Wan2.2
pip install -r requirements.txt

# 2. 下载模型权重
huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir ./Wan2.2-TI2V-5B

# 3. 安装 API 服务依赖
cd /path/to/wan-api
pip install -r requirements.txt
```

## 启动

```bash
python main.py
```

或使用 uvicorn：

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

启动后访问：

| 地址 | 说明 |
|---|---|
| `http://localhost:8000/app` | Gradio 界面 |
| `http://localhost:8000/docs` | Swagger API 文档 |
| `http://localhost:8000/health` | 健康检查 |

## API

### 提交任务

```bash
# 文生视频
curl -X POST http://localhost:8000/api/v1/tasks \
  -H "Content-Type: application/json" \
  -d '{"task_type": "t2v", "prompt": "一只猫在海边冲浪"}'

# 参考图生视频
curl -X POST http://localhost:8000/api/v1/tasks \
  -H "Content-Type: application/json" \
  -d '{"task_type": "i2v", "prompt": "猫在奔跑", "reference_image": "<base64>"}'

# 首尾帧生视频
curl -X POST http://localhost:8000/api/v1/tasks \
  -H "Content-Type: application/json" \
  -d '{"task_type": "ti2v", "prompt": "猫从左走到右", "first_frame": "<base64>", "last_frame": "<base64>"}'
```

### 查询状态

```bash
curl http://localhost:8000/api/v1/tasks/{task_id}
```

### 下载视频

```bash
curl -o output.mp4 http://localhost:8000/api/v1/tasks/{task_id}/video
```

### 取消任务

```bash
curl -X DELETE http://localhost:8000/api/v1/tasks/{task_id}
```

### 队列状态

```bash
curl http://localhost:8000/api/v1/queue/status
```

### WebSocket 实时推送

```javascript
const ws = new WebSocket("ws://localhost:8000/ws/tasks/{task_id}");
ws.onmessage = (e) => console.log(JSON.parse(e.data));
```

## 配置

通过环境变量配置，前缀 `WAN_`：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `WAN_MODEL_CHECKPOINT_DIR` | `./Wan2.2-TI2V-5B` | 模型权重路径 |
| `WAN_DEVICE_ID` | `0` | GPU 设备 ID |
| `WAN_OFFLOAD_MODEL` | `true` | 模型 CPU 卸载（降低显存占用） |
| `WAN_DEFAULT_SIZE` | `1280*704` | 默认分辨率 |
| `WAN_DEFAULT_FRAME_NUM` | `121` | 默认帧数 |
| `WAN_DEFAULT_SAMPLE_STEPS` | `50` | 默认采样步数 |
| `WAN_DEFAULT_GUIDE_SCALE` | `5.0` | 默认引导强度 |
| `WAN_MAX_QUEUE_SIZE` | `100` | 最大队列长度 |
| `WAN_OUTPUT_DIR` | `./outputs` | 视频输出目录 |
| `WAN_HOST` | `0.0.0.0` | 监听地址 |
| `WAN_PORT` | `8000` | 监听端口 |

## 项目结构

```
wan-api/
├── main.py            # FastAPI 入口，路由，WebSocket
├── config.py          # 配置管理
├── schemas.py         # 数据模型
├── queue_manager.py   # 队列抽象接口 + 内存实现
├── worker.py          # GPU Worker，调用 Wan2.2 模型
├── web_ui.py          # Gradio 界面
└── requirements.txt   # 依赖
```

## 替换队列实现

队列接口已抽象为 `BaseQueue`，替换为 Redis 等只需实现该接口：

```python
from queue_manager import BaseQueue

class RedisQueue(BaseQueue):
    async def put(self, task): ...
    async def get(self) -> TaskRecord: ...
    async def cancel(self, task_id) -> bool: ...
    # ...
```

然后在 `main.py` 中替换 `InMemoryQueue` 即可。

## 协议

本项目基于 [Apache License 2.0](LICENSE) 开源。

本项目使用了 [Wan2.2](https://github.com/Wan-Video/Wan2.2) 模型，请同时遵守其开源协议。
