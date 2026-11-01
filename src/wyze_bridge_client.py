"""Direct HTTP client for docker-wyze-bridge PTZ control (Wyze Cam Pan v3 `HL_PAN3` verified)."""

import time
import requests
from src.config import Config


class WyzeBridgeClient:
    """
    docker-wyze-bridge の REST API を叩いて Wyze Cam Pan v3 (HL_PAN3) の向きを制御するクライアント。
    実機検証済みの `ptz_position/<vertical>,<horizontal>` (K11018SetPTZPosition) および
    `rotary_action/<h_dir>,<v_dir>` (K11002SetRotaryByAction) を使用する。
    """

    def __init__(
        self,
        bridge_host: str = Config.WYZE_BRIDGE_HOST,
        cam_name: str = Config.WYZE_CAM_NAME,
        method: str = Config.WYZE_PTZ_METHOD,
        dry_run: bool = False,
        min_interval: float = 0.35,
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
        """
        目標Pan/Tilt角度（正面=0,0）へ Wyze Cam Pan v3 を向ける。
        Pan v3 のハードウェア座標系:
          - horizontal: 0..350 (正面 = 180)
          - vertical  : 0..180 (水平 = 90)
          - エンドポイント: /api/<cam>/ptz_position/<vertical>,<horizontal>
        """
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
            h_code = 0
            v_code = 0
            if abs(err_pan) >= self.deadband_deg:
                h_code = 2 if err_pan > 0 else 1
            if abs(err_tilt) >= self.deadband_deg:
                v_code = 1 if err_tilt > 0 else 2
            url = f"{self.bridge_host}/api/{self.cam_name}/rotary_action/{h_code},{v_code}"
            ok = self._request(url)
            if ok:
                self.current_pan = pan_deg
                self.current_tilt = tilt_deg
            return ok
        else:
            wyze_horiz = int(round(max(0.0, min(350.0, 180.0 + pan_deg))))
            wyze_vert = int(round(max(0.0, min(180.0, 90.0 + tilt_deg))))
            url = f"{self.bridge_host}/api/{self.cam_name}/ptz_position/{wyze_vert},{wyze_horiz}"
            ok = self._request(url)
            if ok:
                self.current_pan = pan_deg
                self.current_tilt = tilt_deg
            return ok

    def send_action(self, action: str) -> bool:
        """left(1,0) / right(2,0) / up(0,1) / down(0,2) / stop(0,0) / reset を送信"""
        action_map = {
            "left": "1,0",
            "right": "2,0",
            "up": "0,1",
            "down": "0,2",
            "stop": "0,0",
        }
        if action == "reset":
            url = f"{self.bridge_host}/api/{self.cam_name}/reset_rotation"
        else:
            payload = action_map.get(action, action)
            url = f"{self.bridge_host}/api/{self.cam_name}/rotary_action/{payload}"
        return self._request(url)

    def _request(self, url: str) -> bool:
        if self.dry_run:
            print(f"[WyzeBridge DRY-RUN] -> {url}")
            return True
        try:
            resp = requests.get(url, timeout=2.0)
            return resp.ok
        except Exception as e:
            print(f"[WyzeBridge WARN] API call failed ({url}): {e}")
            return False
