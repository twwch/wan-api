from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Model
    model_checkpoint_dir: str = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    device_id: int = 0
    offload_model: bool = True

    # Generation defaults
    default_size: str = "1280*704"
    default_frame_num: int = 121
    default_sample_steps: int = 50
    default_sample_shift: float = 5.0
    default_guide_scale: float = 5.0
    default_seed: int = -1  # -1 means random

    # Queue
    max_queue_size: int = 100

    # Storage
    output_dir: str = "./outputs"
    upload_dir: str = "./uploads"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    model_config = {"env_prefix": "WAN_"}


settings = Settings()

# Ensure directories exist
Path(settings.output_dir).mkdir(parents=True, exist_ok=True)
Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
