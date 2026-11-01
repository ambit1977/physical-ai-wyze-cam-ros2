#!/usr/bin/env python3
"""
Quest HMD / Controller × ROS2 × Wyze Cam Pan v3 統合テレオペ & AI制御ノード。

『Quest_テレオペ開発・再利用ガイド.md』の設計（ZeroMQ HWM=1 最新フレーム優先、UDP 8125 自動探索、
クラッチ/ゼロ点再基準化、デッドゾーン・低域フィルタ・Watchdog安全設計）に完全準拠し、
以下の【2つのQuest接続方式】に同時対応します：

方式A: Quest Browser (WebXR HMDヘッドトラッキング + 視界固定シネマスクリーン)
  - URL: https://<MacのIP>:8443 を Quest Browser で開いて「VR HMDモード開始」をタップ
  - HMD（頭）を上下左右どこに向けても、VR空間内で視界正面に固定された大型スクリーンで
    常に Wyze Cam Pan v3 のライブ映像を見ながら、頭の角度(Yaw/Pitch)でカメラを直感操作！

方式B: 既存サイドロード済みAPK『Open Teach LIBERO Controller』(ZeroMQ プロトコル互換)
  - Quest の「提供元不明」から『Open Teach LIBERO Controller』を起動するだけで自動接続！
  - TCP 10505 (メイン画面) / 10506 (サブ画面) に Wyze Cam Pan v3 のライブ映像を配信
  - TCP 8087 (6DoF姿勢) から Yaw / Pitch 角度を抽出し、ROS2 経由で Wyze Cam を駆動
  - Aボタン: 追従の一時停止/再開＋現在姿勢でゼロ点再基準化 (Clutch)
  - Bボタン: AIターゲット自動追従モード (YOLOv8 / VLM) のON/OFF・対象切替
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
from typing import Optional

import cv2
import numpy as np
import zmq

from src.config import Config
from src.camera_stream import CameraStream
from src.ros2_bus import ROS2WyzeBus
from src.models.vlm_target_detector import VLMTargetDetector


WEB_HTML_PATH = Path(__file__).resolve().parent / "web" / "quest_hmd_wyze.html"
CERT_FILE = Path("/tmp/pai_quest_cert.pem")
KEY_FILE = Path("/tmp/pai_quest_key.pem")
AI_TARGETS = ["cup", "bottle", "cell phone", "person"]


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
    """WebXR HMD / OpenTeach ZMQ / AI Detector / メイン制御ループ間で共有する状態"""

    def __init__(self):
        self.lock = threading.Lock()
        self.source_mode: str = "Waiting for Quest..."  # "Quest WebXR HMD" or "Quest OpenTeach APK"
        self.raw_yaw: float = 0.0
        self.raw_pitch: float = 0.0
        self.smooth_pan: float = 0.0
        self.smooth_tilt: float = 0.0
        self.paused: bool = False
        self.ai_mode: bool = False
        self.ai_target_idx: int = 0
        self.last_input_ts: float = 0.0
        self.latest_jpeg: bytes = b""
        self.running: bool = True

    @property
    def current_ai_target(self) -> str:
        return AI_TARGETS[self.ai_target_idx % len(AI_TARGETS)]


def start_udp_discovery_responder(state: SharedTeleopState, port: int = 8125):
    """
    Open Teach LIBERO Controller APK からの UDP 8125 探索パケット ('discover', 'start') に応答する。
    既存の LaunchAgent が動いている場合は一時停止してポート競合を防ぐ。
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
    Open Teach LIBERO Controller APK から送信される TCP 8087 (24 keypoints) と
    TCP 8100 (Bボタン Reset) を受信し、基準姿勢からの相対 Yaw / Pitch 角度に変換する。
    """

    def _extract_yaw_pitch_buttons(token_bytes: bytes):
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

        # GestureDetector.cs CreateControllerSkeleton の幾何構造から前方ベクトルを復元
        origin = arr[0]
        v6 = arr[6] - origin
        v16 = arr[16] - origin
        forward = v6 + v16
        norm = np.linalg.norm(forward)
        if norm < 1e-6:
            return None
        forward = forward / norm

        # Unity座標系 (+X right, +Y up, +Z forward) から Yaw / Pitch (度) を算出
        yaw_deg = float(np.degrees(np.arctan2(forward[0], forward[2])))
        pitch_deg = float(np.degrees(np.arcsin(np.clip(forward[1], -1.0, 1.0))))

        # ボタン判定: points[19] (thumbTip) と points[21] (A button) / points[23] (Trigger) の距離
        a_pressed = float(np.linalg.norm(arr[21] - arr[19])) < 0.015
        trig_pressed = float(np.linalg.norm(arr[23] - arr[19])) < 0.015
        return yaw_deg, pitch_deg, a_pressed, trig_pressed

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

        ref_yaw: Optional[float] = None
        ref_pitch: Optional[float] = None
        prev_a = False
        paused = False

        print(f"[Quest OpenTeach ZMQ] Listening on tcp://0.0.0.0:{right_port} (pose) & {reset_port} (B-button)")
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
                parsed = _extract_yaw_pitch_buttons(raw)
                if parsed is None:
                    continue
                yaw_deg, pitch_deg, a_pressed, trig_pressed = parsed

                # Aボタン押下エッジで Pause 切替 & 再開時に現在姿勢をゼロ点(正面)に再基準化
                if a_pressed and not prev_a:
                    paused = not paused
                    if not paused:
                        ref_yaw = yaw_deg
                        ref_pitch = pitch_deg
                    print(f"[Quest A-Button] Clutch Paused={paused}")
                prev_a = a_pressed

                if ref_yaw is None or trig_pressed:
                    ref_yaw = yaw_deg
                    ref_pitch = pitch_deg

                rel_yaw = ((yaw_deg - ref_yaw + 180.0) % 360.0) - 180.0
                rel_pitch = float(np.clip(pitch_deg - ref_pitch, -60.0, 60.0))

                with state.lock:
                    state.source_mode = "Quest OpenTeach APK (ZMQ)"
                    state.paused = paused
                    if not paused:
                        state.raw_yaw = rel_yaw * 1.4  # コントローラー回転倍率
                        state.raw_pitch = rel_pitch * 1.3
                    state.last_input_ts = time.time()

        pull_right.close()
        pull_reset.close()
        ctx.term()

    t = threading.Thread(target=_loop, daemon=True)
    t.start()


def start_webxr_https_server(state: SharedTeleopState, host_ip: str, port: int = 8443):
    """Quest Browser 用の WebXR HMD サーバー (HTTPS) をバックグラウンド起動"""
    ensure_self_signed_cert(host_ip)
    html_bytes = WEB_HTML_PATH.read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass  # ログ抑制

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
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
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
                        elif data.get("toggle_ai"):
                            state.ai_mode = bool(data.get("ai_mode", not state.ai_mode))
                        else:
                            state.paused = bool(data.get("paused", False))
                            if not state.paused:
                                state.raw_yaw = float(data.get("yaw", 0.0))
                                state.raw_pitch = float(data.get("pitch", 0.0))
                        state.last_input_ts = time.time()
                except Exception:
                    pass
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"ok":true}')
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
    parser = argparse.ArgumentParser(description="Quest HMD/Controller -> ROS2 -> Wyze Cam Pan v3 Teleop")
    parser.add_argument("--wyze-rtsp", type=str, default=Config.WYZE_CAM_RTSP, help="Wyze Cam RTSP URL")
    parser.add_argument("--ai-backend", type=str, default="yolo", choices=["yolo", "owlv2", "moondream"])
    parser.add_argument("--deadzone", type=float, default=1.5, help="HMD tremor deadzone in degrees")
    parser.add_argument("--ema", type=float, default=0.45, help="Low-pass filter smoothing factor")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--record", type=str, default="", help="Optional MP4 path to record demo video")
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

    # 3. Wyze Cam RTSP ストリーム & ROS2 バス & AI 検出器
    wyze_stream = CameraStream(args.wyze_rtsp, width=640, height=480)
    bus = ROS2WyzeBus(dry_run=args.dry_run)
    detector: Optional[VLMTargetDetector] = None

    writer = None

    print("\n" + "=" * 72)
    print("  Quest HMD × ROS2 × Wyze Cam Pan v3 システム起動完了！")
    print(f"  [方式1: HMD頭追従 + 視界固定スクリーン] Questブラウザで開く: https://{lan_ip}:8443")
    print("  [方式2: Open Teach APK] Questの『Open Teach LIBERO Controller』を起動")
    print("  Mac側キー操作: [a] AI自動ロックオン切替 | [t] ターゲット変更 | [r] 正面リセット | [q] 終了")
    print("=" * 72 + "\n")

    try:
        while True:
            frame = wyze_stream.read()
            if frame is None:
                view = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(
                    view,
                    "Waiting for Wyze Cam RTSP stream...",
                    (110, 240),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 200, 255),
                    2,
                )
            else:
                view = frame.copy()

            h, w = view.shape[:2]

            with state.lock:
                ai_mode = state.ai_mode
                target_name = state.current_ai_target
                paused = state.paused
                raw_yaw = state.raw_yaw
                raw_pitch = state.raw_pitch
                source_mode = state.source_mode

            # AI ターゲット自動追従モードがONの場合
            if ai_mode:
                if detector is None:
                    detector = VLMTargetDetector(backend=args.ai_backend)
                det = detector.detect(view, target_name)
                if det is not None:
                    x, y, bw, bh = det.bbox
                    cx_norm, cy_norm = det.center_norm
                    cv2.rectangle(view, (x, y), (x + bw, y + bh), (0, 255, 0), 3)
                    cv2.putText(
                        view,
                        f"AI LOCK: {det.label} ({det.confidence:.2f})",
                        (x, max(24, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 0),
                        2,
                    )
                    with state.lock:
                        state.smooth_pan = float(
                            np.clip(state.smooth_pan + cx_norm * 35.0, Config.PAN_MIN, Config.PAN_MAX)
                        )
                        state.smooth_tilt = float(
                            np.clip(state.smooth_tilt - cy_norm * 25.0, Config.TILT_MIN, Config.TILT_MAX)
                        )
            else:
                # Quest HMD / Controller 角度追従モード (Deadzone + EMAフィルタ)
                with state.lock:
                    err_pan = raw_yaw - state.smooth_pan
                    err_tilt = raw_pitch - state.smooth_tilt
                    if abs(err_pan) >= args.deadzone:
                        state.smooth_pan = (1.0 - args.ema) * state.smooth_pan + args.ema * raw_yaw
                    if abs(err_tilt) >= args.deadzone:
                        state.smooth_tilt = (1.0 - args.ema) * state.smooth_tilt + args.ema * raw_pitch

            with state.lock:
                cmd_pan = state.smooth_pan
                cmd_tilt = state.smooth_tilt

            if not paused:
                bus.publish_ptz(cmd_pan, cmd_tilt)

            # 画面中央クロスヘアとHUD描画 (Quest VR画面にもそのまま転送される！)
            cx, cy = w // 2, h // 2
            cv2.line(view, (cx - 25, cy), (cx + 25, cy), (0, 255, 100), 1)
            cv2.line(view, (cx, cy - 25), (cx, cy + 25), (0, 255, 100), 1)

            cv2.rectangle(view, (10, 10), (455, 138), (15, 20, 30), -1)
            status_label = (
                f"AI AUTO-LOCK ({target_name})"
                if ai_mode
                else ("PAUSED (Press A / Grip)" if paused else "ACTIVE TRACKING")
            )
            status_color = (255, 200, 0) if ai_mode else ((0, 180, 255) if paused else (50, 255, 100))

            cv2.putText(view, f"Input: {source_mode}", (22, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (0, 255, 255), 2)
            cv2.putText(view, f"Mode : {status_label}", (22, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.56, status_color, 2)
            cv2.putText(
                view,
                f"Quest Yaw/Pitch: {raw_yaw:+6.1f} / {raw_pitch:+5.1f} deg",
                (22, 96),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.54,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                view,
                f"ROS2 Wyze PTZ  : Pan {cmd_pan:+6.1f} / Tilt {cmd_tilt:+5.1f} deg",
                (22, 124),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (150, 255, 150),
                2,
            )

            # 1. WebXR 向け JPEG 更新
            ok_jpg, buf_jpg = cv2.imencode(".jpg", view, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            if ok_jpg:
                with state.lock:
                    state.latest_jpeg = buf_jpg.tobytes()

            # 2. OpenTeach APK 向け ZeroMQ 映像配信 (256x256 @ TCP 10505, 128x128 @ TCP 10506)
            # Unity Texture2D.LoadImage 用に上下反転補正した正方形フレームを送信
            sq_main = cv2.resize(view, (256, 256))
            ok_m, buf_m = cv2.imencode(".jpg", sq_main, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            if ok_m:
                pub_main_img.send(buf_m.tobytes())

            # サブ画面 (10506) には中心部の2倍デジタルズーム映像を表示
            crop_x0, crop_y0 = w // 4, h // 4
            zoom_crop = view[crop_y0 : crop_y0 + h // 2, crop_x0 : crop_x0 + w // 2]
            sq_sub = cv2.resize(zoom_crop, (128, 128))
            # HandCameraOverlay は uvRect(1,1,-1,-1) が設定されているため上下左右反転して送ると正立する
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
            elif key == ord("t"):
                with state.lock:
                    state.ai_target_idx = (state.ai_target_idx + 1) % len(AI_TARGETS)
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
