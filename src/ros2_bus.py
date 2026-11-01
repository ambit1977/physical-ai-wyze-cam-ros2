"""
ROS2 Bridge Client (via roslibpy WebSocket) with automatic direct-bridge fallback.
Host-side AI nodes publish `/wyze/cmd_ptz` and `/cmd_vel` to ROS2.
If ROS2 container is not running, it seamlessly falls back to calling WyzeBridgeClient directly.
"""

import json
import time
from typing import Optional
from src.config import Config
from src.wyze_bridge_client import WyzeBridgeClient


class ROS2WyzeBus:
    """
    ROS2 (rosbridge_server) を介して `/wyze/cmd_ptz` および `/cmd_vel` (Twist) をPublishするクラス。
    """

    def __init__(
        self,
        host: str = Config.ROSBRIDGE_HOST,
        port: int = Config.ROSBRIDGE_PORT,
        direct_fallback: bool = True,
        dry_run: bool = False,
    ):
        self.host = host
        self.port = port
        self.dry_run = dry_run
        self.ros_client = None
        self.pub_ptz = None
        self.pub_twist = None
        self.connected = False

        self.fallback_client: Optional[WyzeBridgeClient] = (
            WyzeBridgeClient(dry_run=dry_run) if direct_fallback else None
        )

        self._try_connect_rosbridge()

    def _try_connect_rosbridge(self):
        try:
            import roslibpy

            self.ros_client = roslibpy.Ros(host=self.host, port=self.port)
            self.ros_client.run(timeout=2.0)
            if self.ros_client.is_connected:
                self.pub_ptz = roslibpy.Topic(
                    self.ros_client, "/wyze/cmd_ptz", "std_msgs/String"
                )
                self.pub_twist = roslibpy.Topic(
                    self.ros_client, "/cmd_vel", "geometry_msgs/Twist"
                )
                self.connected = True
                print(f"[ROS2Bus] Connected to rosbridge at ws://{self.host}:{self.port}")
        except Exception as e:
            self.connected = False
            print(
                f"[ROS2Bus] rosbridge (ws://{self.host}:{self.port}) not reachable ({e}). "
                f"Using direct WyzeBridgeClient fallback."
            )

    def publish_ptz(self, pan_deg: float, tilt_deg: float, mode: str = "degree"):
        """目標Pan/Tilt角度をROS2トピック `/wyze/cmd_ptz` にPublish"""
        payload = {
            "pan": round(float(pan_deg), 2),
            "tilt": round(float(tilt_deg), 2),
            "mode": mode,
            "ts": time.time(),
        }
        if self.connected and self.pub_ptz is not None:
            import roslibpy

            self.pub_ptz.publish(roslibpy.Message({"data": json.dumps(payload)}))

        if self.fallback_client is not None and not self.connected:
            self.fallback_client.set_angles(pan_deg, tilt_deg)

    def publish_twist(self, linear_x: float, angular_z: float):
        """ルンバ互換 `/cmd_vel` トピックへ Twist メッセージをPublish"""
        if self.connected and self.pub_twist is not None:
            import roslibpy

            self.pub_twist.publish(
                roslibpy.Message(
                    {
                        "linear": {"x": float(linear_x), "y": 0.0, "z": 0.0},
                        "angular": {"x": 0.0, "y": 0.0, "z": float(angular_z)},
                    }
                )
            )

    def close(self):
        if self.ros_client is not None and self.connected:
            try:
                self.ros_client.terminate()
            except Exception:
                pass
