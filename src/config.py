"""Configuration loader for Wyze Cam ROS2 system."""

import os
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")


class Config:
    WYZE_BRIDGE_HOST: str = os.getenv("WYZE_BRIDGE_HOST", "http://localhost:5050").rstrip("/")
    WYZE_CAM_NAME: str = os.getenv("WYZE_CAM_NAME", "wyze")
    WYZE_PTZ_METHOD: str = os.getenv("WYZE_PTZ_METHOD", "degree")

    # 静止カメラ（ATOM Cam 2 RTSP または USB/Mac内蔵カメラ番号）
    STATIC_CAM_SOURCE: str = os.getenv("STATIC_CAM_SOURCE", "0")
    # Wyze Cam RTSP URL
    WYZE_CAM_RTSP: str = os.getenv("WYZE_CAM_RTSP", f"rtsp://localhost:8554/{WYZE_CAM_NAME}")

    # ROS2 rosbridge 設定
    ROSBRIDGE_HOST: str = os.getenv("ROSBRIDGE_HOST", "localhost")
    ROSBRIDGE_PORT: int = int(os.getenv("ROSBRIDGE_PORT", "9090"))

    # PTZ 可動域 (度)
    PAN_MIN: float = -170.0
    PAN_MAX: float = 170.0
    TILT_MIN: float = -40.0
    TILT_MAX: float = 80.0
