#!/usr/bin/env python3
"""
Quest HMD / 6DoF Arm Controller × ROS2 × Wyze Cam Pan v3 統合テレオペ & AI視線制御ノード。

【実装機能】
1. 多軸アーム動作 → カメラPT軸・視線ベクトル変換 (Cross-Embodiment Arm-to-PTZ Gaze Retargeting):
   - 6DoF/7DoF ロボットアーム動作 `[dx, dy, dz, droll, dpitch, dyaw, gripper]` (Open-Teach / LIBERO / VLA等)
     を球面ヤコビアン投影によりカメラの 3次元視線単位ベクトル `v_gaze = (gx, gy, gz)` および
     Pan/Tilt 角度へリアルタイム変換 (`ArmToPTZGazeRetargeter`)。
2. 暴走防止 Step-and-Settle 視覚サーボ & 人間らしい3D空間見回し探索:
   - 視界に対象物（カップ・ペットボトル・スマホ・牛乳パック）が入っている場合:
     RTSP映像遅延(約0.6秒)による積分ワインドアップ暴走を防ぐため、1ステップ最大 ±11° 動かすごとに
     0.72秒待機して映像の整定を確認しながら対象物を画面中央クロスヘアへピタッとロック。
   - 視界に対象物が入っていない場合:
     人間が周囲を見回すように 3D 作業空間ウェイポイントへ視線ベクトルを巡回させ、各角度で
     1.45秒止まって対象物がないか探索。発見した瞬間に中央ロックへ遷移。
3. Quest デモ記録 & 模倣学習 (Behavior Cloning: `QuestDemoBCManager`):
   - `[d]` キー（またはQuestブラウザUIボタン）で人間がQuestで対象物を探して中央に捉えるデモを記録。
   - `[l]` キーで記録したデモからニューラル方策 (`BCSearchNet`) を即座に学習し、AI探索・追尾に反映。
"""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import socket
import ssl
import subprocess
import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np
import zmq

from src.config import Config
from src.camera_stream import CameraStream
from src.ros2_bus import ROS2WyzeBus
from src.models.vlm_target_detector import VLMTargetDetector
from src.models.arm_to_ptz_retargeter import ArmToPTZGazeRetargeter, GazePTZCommand
from src.models.bc_search_policy import QuestDemoBCManager


WEB_HTML_PATH = Path(__file__).resolve().parent / "web" / "quest_hmd_wyze.html"
CERT_FILE = Path("/tmp/pai_quest_cert.pem")
KEY_FILE = Path("/tmp/pai_quest_key.pem")
# カップ、ペットボトル、スマホ、牛乳パック (YOLO-World / OWLv2 対応)
AI_TARGETS = ["cup", "plastic bottle", "cell phone", "milk carton"]
AI_TARGET_LABELS = ["Cup (カップ)", "Bottle (ペットボトル)", "Phone (スマホ)", "Milk Carton (牛乳パック)"]


def get_lan_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"


def ensure_self_signed_cert(ip_addr: str):
    if CERT_FILE.exists() and KEY_FILE.exists():
        return
    cmd = [
        "openssl",
        "req",
        "-x509",
        "-newkey",
        "rsa:2048",
        "-keyout",
        str(KEY_FILE),
        "-out",
        str(CERT_FILE),
        "-days",
        "365",
        "-nodes",
        "-subj",
        f"/CN={ip_addr}",
    ]
    subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


class SharedTeleopState:
    """WebXR HMD / OpenTeach 6DoF Arm ZMQ / AI Detector / メイン制御ループ間で共有する状態"""

    def __init__(self):
        self.lock = threading.Lock()
        self.source_mode: str = "Waiting for Quest..."
        self.raw_yaw: float = 0.0
        self.raw_pitch: float = 0.0
        self.smooth_pan: float = 0.0
        self.smooth_tilt: float = 0.0
        self.gaze_vec: Tuple[float, float, float] = (0.0, 0.0, 1.0)
        self.virtual_ee_pos: Tuple[float, float, float] = (0.0, 0.0, 0.65)
        self.foveation_zoom: float = 1.0
        self.ai_phase: str = "IDLE"
        self.paused: bool = False
        self.ai_mode: bool = False
        self.ai_target_idx: int = 0
        self.req_toggle_record: bool = False
        self.req_train_bc: bool = False
        self.last_input_ts: float = 0.0
        self.latest_jpeg: bytes = b""
        self.running: bool = True

    @property
    def current_ai_target(self) -> str:
        return AI_TARGETS[self.ai_target_idx % len(AI_TARGETS)]


def start_udp_discovery_responder(state: SharedTeleopState, port: int = 8125):
    """
    Open Teach LIBERO Controller APK からの UDP 8125 探索パケット ('discover', 'start') に応答する。
    """
    uid = subprocess.getoutput("id -u").strip()
    subprocess.run(
        ["launchctl", "bootout", f"gui/{uid}/com.openteach.quest-launcher"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    def _loop():
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("0.0.0.0", port))
            sock.settimeout(0.5)
            print(f"[Quest UDP Discovery] Listening on udp://0.0.0.0:{port}")
        except Exception as e:
            print(f"[Quest UDP Discovery] Could not bind UDP {port}: {e}")
            return

        while state.running:
            try:
                payload, addr = sock.recvfrom(4096)
                req = json.loads(payload.decode("utf-8"))
                req_id = str(req.get("id", "0"))
                cmd = str(req.get("command", "")).lower()
                if cmd == "discover":
                    resp = json.dumps(
                        {"id": req_id, "status": "discovered", "stage": 3},
                        separators=(",", ":"),
                    ).encode("utf-8")
                    sock.sendto(resp, addr)
                    print(f"[Quest UDP Discovery] Answered Quest at {addr[0]}")
            except socket.timeout:
                continue
            except Exception:
                continue
        sock.close()

    t = threading.Thread(target=_loop, daemon=True)
    t.start()


def start_openteach_zmq_receiver(
    state: SharedTeleopState,
    right_port: int = 8087,
    reset_port: int = 8100,
):
    """
    Open Teach LIBERO Controller APK (多軸アームテレオペ用 6DoF コントローラ) から送信される
    TCP 8087 (24 keypoints) と TCP 8100 (Bボタン Reset) を受信し、
    多軸アームの手先 3D 変位 `(dx, dy, dz)` + 手首回転 `(dyaw, dpitch)` を
    球面視線投影によってカメラの PT 軸角度と 3D 視線ベクトルに変換する。
    """

    def _extract_6dof_arm_pose_and_buttons(token_bytes: bytes):
        text = token_bytes.decode("utf-8", errors="ignore").strip()
        if ":" not in text:
            return None
        vec_strs = text.split(":", 1)[1].strip().split("|")
        pts = []
        for vs in vec_strs:
            coords = [float(x) for x in vs.split(",")[:3]]
            pts.append(coords)
        if len(pts) < 24:
            return None
        arr = np.asarray(pts, dtype=np.float64)

        # 手先3D座標 origin = arr[0] (Unity座標系: +X right, +Y up, +Z forward [m])
        origin = arr[0]
        v6 = arr[6] - origin
        v16 = arr[16] - origin
        forward = v6 + v16
        norm = np.linalg.norm(forward)
        if norm < 1e-6:
            return None
        forward = forward / norm

        yaw_deg = float(np.degrees(np.arctan2(forward[0], forward[2])))
        pitch_deg = float(np.degrees(np.arcsin(np.clip(forward[1], -1.0, 1.0))))

        a_pressed = float(np.linalg.norm(arr[21] - arr[19])) < 0.015
        trig_pressed = float(np.linalg.norm(arr[23] - arr[19])) < 0.015
        return origin, yaw_deg, pitch_deg, a_pressed, trig_pressed

    def _loop():
        ctx = zmq.Context()
        pull_right = ctx.socket(zmq.PULL)
        pull_right.setsockopt(zmq.LINGER, 0)
        pull_right.setsockopt(zmq.RCVHWM, 2)
        pull_right.bind(f"tcp://0.0.0.0:{right_port}")

        pull_reset = ctx.socket(zmq.PULL)
        pull_reset.setsockopt(zmq.LINGER, 0)
        pull_reset.bind(f"tcp://0.0.0.0:{reset_port}")

        poller = zmq.Poller()
        poller.register(pull_right, zmq.POLLIN)
        poller.register(pull_reset, zmq.POLLIN)

        ref_origin: Optional[np.ndarray] = None
        ref_yaw: Optional[float] = None
        ref_pitch: Optional[float] = None
        prev_a = False
        paused = False

        print(f"[Quest OpenTeach 6DoF->PTZ] Listening on tcp://0.0.0.0:{right_port} (6DoF arm) & {reset_port} (B-btn)")
        while state.running:
            events = dict(poller.poll(100))
            if pull_reset in events:
                msg = pull_reset.recv()
                if msg == b"Reset":
                    with state.lock:
                        if not state.ai_mode:
                            state.ai_mode = True
                        else:
                            state.ai_target_idx += 1
                            if state.ai_target_idx >= len(AI_TARGETS):
                                state.ai_target_idx = 0
                                state.ai_mode = False
                        print(
                            f"[Quest B-Button] AI Mode={state.ai_mode}, Target='{state.current_ai_target}'"
                        )

            if pull_right in events:
                raw = pull_right.recv()
                parsed = _extract_6dof_arm_pose_and_buttons(raw)
                if parsed is None:
                    continue
                origin, yaw_deg, pitch_deg, a_pressed, trig_pressed = parsed

                if a_pressed and not prev_a:
                    paused = not paused
                    if not paused:
                        ref_origin = origin.copy()
                        ref_yaw = yaw_deg
                        ref_pitch = pitch_deg
                    print(f"[Quest A-Button] Clutch Paused={paused}")
                prev_a = a_pressed

                if ref_yaw is None or ref_origin is None or trig_pressed:
                    ref_origin = origin.copy()
                    ref_yaw = yaw_deg
                    ref_pitch = pitch_deg

                # 多軸アームの手先 3D 変位 (dx, dy, dz) [m] と手首回転 (rel_yaw, rel_pitch) [deg]
                dpos = origin - ref_origin
                dx, dy, dz = float(dpos[0]), float(dpos[1]), float(dpos[2])
                rel_yaw = ((yaw_deg - ref_yaw + 180.0) % 360.0) - 180.0
                rel_pitch = float(np.clip(pitch_deg - ref_pitch, -60.0, 60.0))

                # 多軸アームの手先位置 + 手首回転を球面視線ベクトルへ統合変換:
                # 手先を左右(dx)・上下(dy)に動かした視線角度 + 手首の向き(rel_yaw, rel_pitch)
                depth_eff = max(0.35, 0.65 + dz)
                arm_trans_pan = math.degrees(math.atan2(dx, depth_eff))
                arm_trans_tilt = math.degrees(math.atan2(dy, math.sqrt(dx * dx + depth_eff * depth_eff)))

                combined_pan = float(np.clip(arm_trans_pan * 0.8 + rel_yaw * 1.15, Config.PAN_MIN, Config.PAN_MAX))
                combined_tilt = float(np.clip(arm_trans_tilt * 0.8 + rel_pitch * 1.15, Config.TILT_MIN, Config.TILT_MAX))

                with state.lock:
                    state.source_mode = "Quest 6DoF Arm->PTZ (ZMQ)"
                    state.paused = paused
                    if not paused:
                        state.raw_yaw = combined_pan
                        state.raw_pitch = combined_tilt
                    state.last_input_ts = time.time()

        pull_right.close()
        pull_reset.close()
        ctx.term()

    t = threading.Thread(target=_loop, daemon=True)
    t.start()


def start_webxr_https_server(state: SharedTeleopState, host_ip: str, port: int = 8443):
    """Quest Browser 用の WebXR HMD サーバー (HTTPS) をバックグラウンド起動"""
    ensure_self_signed_cert(host_ip)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def do_GET(self):
            if self.path.startswith("/frame.jpg"):
                with state.lock:
                    jpg = state.latest_jpeg
                if not jpg:
                    self.send_response(204)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
                self.send_header("Content-Length", str(len(jpg)))
                self.end_headers()
                self.wfile.write(jpg)
            else:
                # 常に最新の quest_hmd_wyze.html を読み込み、ブラウザキャッシュを無効化して返す
                html_bytes = WEB_HTML_PATH.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
                self.send_header("Content-Length", str(len(html_bytes)))
                self.end_headers()
                self.wfile.write(html_bytes)

        def do_POST(self):
            if self.path.startswith("/hmd_pose"):
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length)
                try:
                    data = json.loads(body.decode("utf-8"))
                    with state.lock:
                        state.source_mode = "Quest WebXR HMD"
                        if data.get("recenter"):
                            state.raw_yaw = 0.0
                            state.raw_pitch = 0.0
                            state.smooth_pan = 0.0
                            state.smooth_tilt = 0.0
                        elif data.get("toggle_record_demo"):
                            state.req_toggle_record = True
                        elif data.get("train_bc"):
                            state.req_train_bc = True
                        elif data.get("set_target"):
                            state.ai_mode = bool(data.get("ai_mode", False))
                            state.ai_target_idx = int(data.get("target_idx", 0)) % len(AI_TARGETS)
                            print(
                                f"[Quest WebXR] AI Mode={state.ai_mode}, Target='{state.current_ai_target}'"
                            )
                        elif data.get("toggle_ai"):
                            state.ai_mode = bool(data.get("ai_mode", not state.ai_mode))
                        else:
                            state.paused = bool(data.get("paused", False))
                            if not state.paused:
                                state.raw_yaw = float(data.get("yaw", 0.0))
                                state.raw_pitch = float(data.get("pitch", 0.0))
                        state.last_input_ts = time.time()
                        resp_payload = {
                            "ok": True,
                            "ai_mode": state.ai_mode,
                            "target_idx": state.ai_target_idx,
                            "paused": state.paused,
                        }
                except Exception:
                    resp_payload = {"ok": False}

                resp_bytes = json.dumps(resp_payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(resp_bytes)))
                self.end_headers()
                self.wfile.write(resp_bytes)
            else:
                self.send_response(404)
                self.end_headers()

    def _serve():
        httpd = ThreadingHTTPServer(("0.0.0.0", port), Handler)
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(certfile=str(CERT_FILE), keyfile=str(KEY_FILE))
        httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)
        print(f"[Quest WebXR HTTPS] Ready at https://{host_ip}:{port}")
        httpd.serve_forever()

    t = threading.Thread(target=_serve, daemon=True)
    t.start()


def main():
    parser = argparse.ArgumentParser(description="Quest HMD/6DoF Arm -> ROS2 -> Wyze Cam Pan v3 Teleop & AI Search")
    parser.add_argument("--wyze-rtsp", type=str, default=Config.WYZE_CAM_RTSP, help="Wyze Cam RTSP URL")
    parser.add_argument("--ai-backend", type=str, default="yolo", choices=["yolo", "owlv2", "moondream"])
    parser.add_argument("--conf", type=float, default=0.35, help="Detection confidence threshold")
    parser.add_argument("--deadzone", type=float, default=1.5, help="HMD tremor deadzone in degrees")
    parser.add_argument("--ema", type=float, default=0.45, help="Low-pass filter smoothing factor")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--record", "--Record", dest="record", type=str, default="", help="Optional MP4 path to record demo video")
    args = parser.parse_args()

    lan_ip = get_lan_ip()
    state = SharedTeleopState()

    # 1. Quest 通信サーバー起動 (WebXR HTTPS + OpenTeach ZMQ + UDP Discovery)
    start_udp_discovery_responder(state, port=8125)
    start_openteach_zmq_receiver(state, right_port=8087, reset_port=8100)
    start_webxr_https_server(state, host_ip=lan_ip, port=8443)

    # 2. OpenTeach APK 向け映像配信用 ZeroMQ PUB ソケット (TCP 10505 メイン / 10506 サブ / 10012 HUD)
    zmq_ctx = zmq.Context()
    pub_main_img = zmq_ctx.socket(zmq.PUB)
    pub_main_img.setsockopt(zmq.SNDHWM, 1)
    pub_main_img.setsockopt(zmq.LINGER, 0)
    pub_main_img.bind("tcp://0.0.0.0:10505")

    pub_sub_img = zmq_ctx.socket(zmq.PUB)
    pub_sub_img.setsockopt(zmq.SNDHWM, 1)
    pub_sub_img.setsockopt(zmq.LINGER, 0)
    pub_sub_img.bind("tcp://0.0.0.0:10506")

    pub_hud_json = zmq_ctx.socket(zmq.PUB)
    pub_hud_json.setsockopt(zmq.SNDHWM, 1)
    pub_hud_json.setsockopt(zmq.LINGER, 0)
    pub_hud_json.bind("tcp://0.0.0.0:10012")

    # 3. Wyze Cam RTSP ストリーム (16:9 広角フル画角 960x540) & ROS2 バス & 多軸アーム→PT視線変換器 & 模倣学習管理
    wyze_stream = CameraStream(args.wyze_rtsp, width=960, height=540)
    bus = ROS2WyzeBus(dry_run=args.dry_run)
    detector: Optional[VLMTargetDetector] = None
    retargeter = ArmToPTZGazeRetargeter()
    bc_manager = QuestDemoBCManager(num_targets=len(AI_TARGETS))

    writer = None

    print("\n" + "=" * 82)
    print("  Quest MR Passthrough × 多軸アーム→PT視線変換 × ROS2 Wyze Cam Pan v3 起動完了！")
    print(f"  [方式1: WebXR MRパススルー] Questブラウザで開く: https://{lan_ip}:8443")
    print("  [方式2: Open Teach APK (MRパススルー + 6DoFアーム姿勢→PT視線変換)] アプリを起動するだけ")
    print("  ターゲット: 1.カップ(cup) / 2.ペットボトル(plastic bottle) / 3.スマホ(cell phone) / 4.牛乳パック(milk carton)")
    print("  Macキー: [a] AI探索・中央ロックON/OFF | [t] 対象切替 | [d] デモ記録開始/停止 | [l] 模倣学習 | [r] 正面 | [q] 終了")
    print("=" * 82 + "\n")

    try:
        while True:
            now = time.time()
            frame = wyze_stream.read()
            if frame is None:
                view = np.zeros((540, 960, 3), dtype=np.uint8)
                cv2.putText(
                    view,
                    "Waiting for Wyze Cam 16:9 RTSP stream...",
                    (220, 270),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 200, 255),
                    2,
                )
            else:
                view = frame.copy()

            h, w = view.shape[:2]

            with state.lock:
                ai_mode = state.ai_mode
                target_idx = state.ai_target_idx
                target_name = state.current_ai_target
                paused = state.paused
                raw_yaw = state.raw_yaw
                raw_pitch = state.raw_pitch
                source_mode = state.source_mode
                req_rec = state.req_toggle_record
                req_train = state.req_train_bc
                state.req_toggle_record = False
                state.req_train_bc = False

            if req_rec:
                bc_manager.toggle_recording(state.smooth_pan, state.smooth_tilt)
            if req_train:
                bc_manager.train_policy()

            # 物体検出 (AIモード中、または人間がQuestでデモ記録中の場合も視覚特徴として実行)
            det = None
            if ai_mode or bc_manager.recording:
                if detector is None:
                    detector = VLMTargetDetector(backend=args.ai_backend)
                det = detector.detect(view, target_name, conf_threshold=args.conf)

            det_center_norm = det.center_norm if det is not None else None
            det_bbox = det.bbox if det is not None else None
            det_area_norm = 0.0
            if det is not None:
                x, y, bw, bh = det.bbox
                det_area_norm = float(bw * bh) / float(max(1, w * h))
                box_color = (0, 255, 0) if ai_mode else (0, 200, 255)
                cv2.rectangle(view, (x, y), (x + bw, y + bh), box_color, 3)
                cv2.putText(
                    view,
                    f"TARGET: {det.label} ({det.confidence:.2f})",
                    (x, max(24, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    box_color,
                    2,
                )
                # 画面中心から対象物への視線誘導ベクトル矢印を描画
                obj_cx = int(x + 0.5 * bw)
                obj_cy = int(y + 0.5 * bh)
                cv2.arrowedLine(view, (w // 2, h // 2), (obj_cx, obj_cy), (0, 255, 255), 2, tipLength=0.15)

            # A. AI ターゲット探索 & 視覚サーボ中央ロックモード (多軸アーム動作 → PT軸・視線変換)
            if ai_mode:
                bc_delta = bc_manager.predict_delta(
                    frame_bgr=frame if frame is not None else view,
                    target_idx=target_idx,
                    det_center_norm=det_center_norm,
                    det_bbox=det_bbox,
                    current_pan=state.smooth_pan,
                    current_tilt=state.smooth_tilt,
                )
                gaze_cmd: GazePTZCommand = retargeter.step_search_and_center(
                    now=now,
                    det_center_norm=det_center_norm,
                    det_bbox_area_norm=det_area_norm,
                    current_pan=state.smooth_pan,
                    current_tilt=state.smooth_tilt,
                    bc_delta=bc_delta,
                )
                with state.lock:
                    state.smooth_pan = gaze_cmd.pan_deg
                    state.smooth_tilt = gaze_cmd.tilt_deg
                    state.gaze_vec = gaze_cmd.gaze_vector
                    state.virtual_ee_pos = gaze_cmd.virtual_ee_pos
                    state.foveation_zoom = gaze_cmd.foveation_zoom
                    state.ai_phase = gaze_cmd.phase
            else:
                # B. Quest HMD / 6DoF Arm Controller 追従モード (Deadzone + EMAフィルタ + 3D視線ベクトル算出)
                with state.lock:
                    err_pan = raw_yaw - state.smooth_pan
                    err_tilt = raw_pitch - state.smooth_tilt
                    if abs(err_pan) >= args.deadzone:
                        state.smooth_pan = (1.0 - args.ema) * state.smooth_pan + args.ema * raw_yaw
                    if abs(err_tilt) >= args.deadzone:
                        state.smooth_tilt = (1.0 - args.ema) * state.smooth_tilt + args.ema * raw_pitch
                    g_vec, ee_xyz = retargeter.ptz_to_gaze_vector(state.smooth_pan, state.smooth_tilt)
                    state.gaze_vec = g_vec
                    state.virtual_ee_pos = ee_xyz
                    state.foveation_zoom = 1.0
                    state.ai_phase = "HUMAN_TELEOP"

                # デモ記録中なら、人間の Quest 操作ステップを学習データとして記録
                if bc_manager.recording and frame is not None:
                    bc_manager.record_step_if_active(
                        frame_bgr=frame,
                        target_idx=target_idx,
                        det_center_norm=det_center_norm,
                        det_bbox=det_bbox,
                        cmd_pan=state.smooth_pan,
                        cmd_tilt=state.smooth_tilt,
                    )

            with state.lock:
                cmd_pan = state.smooth_pan
                cmd_tilt = state.smooth_tilt
                gaze_vec = state.gaze_vec
                ee_pos = state.virtual_ee_pos
                fov_zoom = state.foveation_zoom
                ai_phase = state.ai_phase

            if not paused:
                bus.publish_ptz(cmd_pan, cmd_tilt)

            # 画面中央クロスヘアと HUD 描画 (Quest 画面にも転送)
            cx, cy = w // 2, h // 2
            cv2.line(view, (cx - 28, cy), (cx + 28, cy), (0, 255, 100), 1)
            cv2.line(view, (cx, cy - 28), (cx, cy + 28), (0, 255, 100), 1)
            cv2.circle(view, (cx, cy), int(0.08 * w), (0, 180, 80), 1)  # 中央ロック許容円

            cv2.rectangle(view, (10, 10), (575, 178), (15, 20, 30), -1)
            status_label = (
                f"AI [{target_name}] -> {ai_phase}"
                if ai_mode
                else ("PAUSED (Press A / Grip)" if paused else "QUEST TELEOP ACTIVE")
            )
            status_color = (50, 220, 255) if ai_mode else ((0, 180, 255) if paused else (50, 255, 100))

            cv2.putText(view, f"Input : {source_mode}", (22, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (0, 255, 255), 2)
            cv2.putText(view, f"Status: {status_label}", (22, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.54, status_color, 2)
            cv2.putText(
                view,
                f"Arm EE->Gaze Vec: EE({ee_pos[0]:+.2f},{ee_pos[1]:+.2f},{ee_pos[2]:+.2f})m | Gaze({gaze_vec[0]:+.2f},{gaze_vec[1]:+.2f},{gaze_vec[2]:+.2f})",
                (22, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.47,
                (220, 220, 255),
                1,
            )
            cv2.putText(
                view,
                f"ROS2 Wyze PTZ   : Pan {cmd_pan:+6.1f} deg / Tilt {cmd_tilt:+5.1f} deg",
                (22, 118),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.54,
                (150, 255, 150),
                2,
            )
            bc_status = (
                f"REC DEMO ({len(bc_manager.states)} steps)"
                if bc_manager.recording
                else (
                    f"BC Policy READY ({bc_manager.episodes_count} eps)"
                    if bc_manager.policy_ready
                    else f"Demos: {len(bc_manager.states)} steps ([d]:Rec [l]:Train)"
                )
            )
            bc_color = (0, 80, 255) if bc_manager.recording else ((180, 255, 100) if bc_manager.policy_ready else (180, 180, 180))
            cv2.putText(
                view,
                f"Imitation (BC)  : {bc_status}",
                (22, 146),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                bc_color,
                2,
            )
            cv2.putText(
                view,
                "Keys: [a]AI Search/Lock [t]Target [d]Rec Demo [l]Train BC [r]Center [q]Quit",
                (22, 168),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                (200, 200, 200),
                1,
            )

            # 1. WebXR 向け JPEG 更新
            ok_jpg, buf_jpg = cv2.imencode(".jpg", view, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            if ok_jpg:
                with state.lock:
                    state.latest_jpeg = buf_jpg.tobytes()

            # 2. OpenTeach APK 向け ZeroMQ 映像配信 (TCP 10505)
            sq_main = np.zeros((256, 256, 3), dtype=np.uint8)
            wide_small = cv2.resize(view, (256, 144))
            sq_main[56:200, 0:256] = wide_small
            cv2.putText(sq_main, f"{status_label[:28]}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, status_color, 1)
            cv2.putText(
                sq_main,
                f"Pan:{cmd_pan:+5.1f} Tilt:{cmd_tilt:+5.1f}",
                (8, 44),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.44,
                (255, 255, 255),
                1,
            )
            cv2.putText(
                sq_main,
                f"Gaze:({gaze_vec[0]:+.2f},{gaze_vec[1]:+.2f},{gaze_vec[2]:+.2f})",
                (8, 220),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (180, 255, 220),
                1,
            )
            cv2.putText(
                sq_main,
                "A:Clutch/Center  B:Target",
                (8, 242),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (180, 220, 255),
                1,
            )
            ok_m, buf_m = cv2.imencode(".jpg", sq_main, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            if ok_m:
                pub_main_img.send(buf_m.tobytes())

            # サブ画面 (10506) にはアーム注視度 (foveation_zoom) に連動した中心部ズーム映像を表示
            zoom_div = max(1.5, min(3.0, 2.0 * fov_zoom))
            zw, zh = int(w / zoom_div), int(h / zoom_div)
            crop_x0 = max(0, (w - zw) // 2)
            crop_y0 = max(0, (h - zh) // 2)
            zoom_crop = view[crop_y0 : crop_y0 + zh, crop_x0 : crop_x0 + zw]
            sq_sub = cv2.resize(zoom_crop, (128, 128))
            sq_sub = cv2.flip(sq_sub, -1)
            ok_s, buf_s = cv2.imencode(".jpg", sq_sub, [int(cv2.IMWRITE_JPEG_QUALITY), 65])
            if ok_s:
                pub_sub_img.send(buf_s.tobytes())

            # HUD JSON (TCP 10012): J1=Pan(rad), J2=Tilt(rad)
            hud_angles = [
                float(np.deg2rad(cmd_pan)),
                float(np.deg2rad(cmd_tilt)),
                0.0,
                -1.5,
                0.0,
                1.8,
                0.0,
            ]
            pub_hud_json.send_string("joint_angles " + json.dumps(hud_angles))

            if args.record:
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(args.record, fourcc, 15.0, (w, h))
                writer.write(view)

            cv2.imshow("Quest HMD x ROS2 x Wyze Cam Pan v3 Monitor", view)
            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("a"):
                with state.lock:
                    state.ai_mode = not state.ai_mode
                    retargeter.last_step_ts = 0.0
                    retargeter.confirm_streak = 0
                    print(f"[Key 'a'] AI Mode={state.ai_mode}, Target='{state.current_ai_target}'")
            elif key == ord("t"):
                with state.lock:
                    state.ai_target_idx = (state.ai_target_idx + 1) % len(AI_TARGETS)
                    retargeter.confirm_streak = 0
                    print(f"[Key 't'] Target switched to '{state.current_ai_target}'")
            elif key == ord("d"):
                bc_manager.toggle_recording(state.smooth_pan, state.smooth_tilt)
            elif key == ord("l"):
                bc_manager.train_policy()
            elif key == ord("r"):
                with state.lock:
                    state.raw_yaw = 0.0
                    state.raw_pitch = 0.0
                    state.smooth_pan = 0.0
                    state.smooth_tilt = 0.0
                bus.publish_ptz(0.0, 0.0)
    finally:
        state.running = False
        if writer is not None:
            writer.release()
        wyze_stream.release()
        bus.close()
        pub_main_img.close()
        pub_sub_img.close()
        pub_hud_json.close()
        zmq_ctx.term()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
