"""Direct HTTP client & simulator for docker-wyze-bridge PTZ control."""

import time
import requests
from src.config import Config


class WyzeBridgeClient:
    """
    docker-wyze-bridge の REST API を叩いて Wyze Cam Pan v3 の向きを制御するクライアント。
    ROS2 プロキシノード経由でも、ホストから直接でも利用可能。
    """

    def __init__(
        self,
        bridge_host: str = Config.WYZE_BRIDGE_HOST,
        cam_name: str = Config.WYZE_CAM_NAME,
        method: str = Config.WYZE_PTZ_METHOD,
        dry_run: bool = False,
        min_interval: float = 0.45,
    ):
        self.bridge_host = bridge_host.rstrip("/")
        self.cam_name = cam_name
        self.method = method
        self.dry_run = dry_run
        self.min_interval = min_interval

        self.current_pan = 0.0
        self.current_tilt = 0.0
        self.last_sent_time = 0.0
        self.deadband_deg = 2.5

    def set_angles(self, pan_deg: float, tilt_deg: float, force: bool = False) -> bool:
        """目標Pan/Tilt角度（正面=0,0）へカメラを向ける"""
        pan_deg = max(Config.PAN_MIN, min(Config.PAN_MAX, float(pan_deg)))
        tilt_deg = max(Config.TILT_MIN, min(Config.TILT_MAX, float(tilt_deg)))

        now = time.time()
        if not force and (now - self.last_sent_time < self.min_interval):
            return False

        err_pan = pan_deg - self.current_pan
        err_tilt = tilt_deg - self.current_tilt
        if not force and abs(err_pan) < self.deadband_deg and abs(err_tilt) < self.deadband_deg:
            return False

        self.last_sent_time = now

        if self.method == "action":
            if abs(err_pan) >= abs(err_tilt):
                direction = "right" if err_pan > 0 else "left"
                ok = self.send_action(direction)
                if ok:
                    self.current_pan += 10.0 if err_pan > 0 else -10.0
                return ok
            else:
                direction = "up" if err_tilt > 0 else "down"
                ok = self.send_action(direction)
                if ok:
                    self.current_tilt += 8.0 if err_tilt > 0 else -8.0
                return ok
        else:
            # docker-wyze-bridge rotary_degree: pan [0..360], tilt [0..90]
            wyze_pan = int(round((pan_deg + 180.0) % 360.0))
            wyze_tilt = int(round(max(0.0, min(90.0, tilt_deg + 45.0))))
            url = f"{self.bridge_host}/api/{self.cam_name}/rotary_degree/{wyze_pan},{wyze_tilt}"
            ok = self._request(url)
            if ok:
                self.current_pan = pan_deg
                self.current_tilt = tilt_deg
            return ok

    def send_action(self, action: str) -> bool:
        """left / right / up / down / stop / reset_rotation を送信"""
        if action == "reset":
            url = f"{self.bridge_host}/api/{self.cam_name}/reset_rotation"
        else:
            url = f"{self.bridge_host}/api/{self.cam_name}/rotary_action/{action}"
        return self._request(url)

    def _request(self, url: str) -> bool:
        if self.dry_run:
            print(f"[WyzeBridge DRY-RUN] -> {url}")
            return True
        try:
            resp = requests.post(url, timeout=2.0)
            if resp.status_code >= 400:
                resp = requests.get(url, timeout=2.0)
            return resp.ok
        except Exception as e:
            print(f"[WyzeBridge WARN] API call failed ({url}): {e}")
            return False
