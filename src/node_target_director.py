#!/usr/bin/env python3
"""
【モード2】指定したもの（自然言語プロンプト・物体名）に Wyze Cam Pan v3 の向きを向けるノード。
Hugging Face のオープンボキャブラリ視覚モデル (`OWLv2` / `Moondream2`) または `YOLOv8` を使用。

動作モード:
  - `--camera-role active` (デフォルト): Wyze Cam自身の映像（またはテスト用カメラ）を見て、
    対象が見つからなければ自動スキャン（首振り探索）し、見つけたら画面中央にロックオン（ビジュアルサーボ追従）
  - `--camera-role static`: 静止カメラ（ATOM Cam 2）の広角映像から対象の位置を見つけ、
    その角度に Wyze Cam Pan v3 を一発で向ける

使い方:
  # Hugging Face OWLv2 (任意の英語テキストで指定可能!)
  python -m src.node_target_director --target "coffee mug" --backend owlv2

  # Hugging Face Moondream2 VLM
  python -m src.node_target_director --target "black bottle" --backend moondream

  # YOLOv8 高速リアルタイム検出 (cup, bottle, cell phone, person, book 等)
  python -m src.node_target_director --target "cup" --backend yolo
"""

import argparse
import math
import time
import cv2
import numpy as np

from src.config import Config
from src.camera_stream import CameraStream
from src.ros2_bus import ROS2WyzeBus
from src.models.vlm_target_detector import VLMTargetDetector


PRESET_TARGETS = ["cup", "bottle", "cell phone", "person", "book", "remote"]


def main():
    parser = argparse.ArgumentParser(description="Mode 2: VLM / Zero-Shot Target Director for Wyze Cam Pan v3")
    parser.add_argument("--cam", type=str, default=Config.STATIC_CAM_SOURCE, help="Camera source (Wyze RTSP, ATOM RTSP, or USB index 0)")
    parser.add_argument("--target", type=str, default="cup", help="Target object description (e.g., 'cup', 'red mug', 'cell phone')")
    parser.add_argument("--backend", type=str, default="owlv2", choices=["owlv2", "moondream", "yolo"], help="AI Vision/VLM backend")
    parser.add_argument("--camera-role", type=str, default="active", choices=["active", "static"], help="'active' (visual servoing) or 'static' (direct angle mapping)")
    parser.add_argument("--fov-h", type=float, default=95.0, help="Horizontal field of view in degrees")
    parser.add_argument("--fov-v", type=float, default=60.0, help="Vertical field of view in degrees")
    parser.add_argument("--dry-run", action="store_true", help="Run without sending HTTP commands to Wyze")
    parser.add_argument("--record", type=str, default="", help="Optional MP4 output file path")
    args = parser.parse_args()

    stream = CameraStream(args.cam)
    detector = VLMTargetDetector(backend=args.backend)
    bus = ROS2WyzeBus(dry_run=args.dry_run)

    target_prompt = args.target
    current_pan = 0.0
    current_tilt = 0.0
    last_seen_time = time.time()
    scan_phase = 0.0
    writer = None

    print(
        f"=== Mode 2: VLM Target Director Started ===\n"
        f"  Backend: {detector.backend} | Target: '{target_prompt}'\n"
        f"  Keys: [1-6] Switch preset targets | [n] Enter custom prompt in terminal | [q] Quit"
    )

    try:
        while True:
            frame = stream.read()
            if frame is None:
                time.sleep(0.02)
                continue

            view = frame.copy()
            h, w = view.shape[:2]

            det = detector.detect(view, target_prompt)
            now = time.time()

            # 画面中央のクロスヘア描画
            cv2.line(view, (w // 2 - 20, h // 2), (w // 2 + 20, h // 2), (180, 180, 180), 1)
            cv2.line(view, (w // 2, h // 2 - 20), (w // 2, h // 2 + 20), (180, 180, 180), 1)

            if det is not None:
                last_seen_time = now
                x, y, bw, bh = det.bbox
                cx_norm, cy_norm = det.center_norm
                cx_px, cy_px = int(x + 0.5 * bw), int(y + 0.5 * bh)

                # 検出ボックスとロックオン線を描画
                cv2.rectangle(view, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
                cv2.line(view, (w // 2, h // 2), (cx_px, cy_px), (0, 255, 255), 2)
                cv2.circle(view, (cx_px, cy_px), 6, (0, 0, 255), -1)
                cv2.putText(
                    view,
                    f"{det.label} ({det.confidence:.2f})",
                    (x, max(20, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

                # カメラ角度計算
                if args.camera_role == "static":
                    # 静止カメラ内の位置から直接Wyze Camの絶対Pan/Tilt角へ変換
                    current_pan = cx_norm * args.fov_h
                    current_tilt = -cy_norm * args.fov_v
                else:
                    # Active Visual Servoing (PID比例制御で画面中央へ収束)
                    kp_pan = 0.45 * args.fov_h
                    kp_tilt = 0.45 * args.fov_v
                    current_pan = float(np.clip(current_pan + cx_norm * kp_pan, Config.PAN_MIN, Config.PAN_MAX))
                    current_tilt = float(np.clip(current_tilt - cy_norm * kp_tilt, Config.TILT_MIN, Config.TILT_MAX))

                bus.publish_ptz(current_pan, current_tilt)
                state_text = "LOCKED ON TARGET"
                state_color = (0, 255, 100)
            else:
                # 3秒以上見失ったら自動スキャン探索
                if now - last_seen_time > 3.0:
                    scan_phase += 0.08
                    current_pan = 65.0 * math.sin(scan_phase)
                    current_tilt = 15.0 * math.cos(scan_phase * 0.5)
                    bus.publish_ptz(current_pan, current_tilt)
                    state_text = "SCANNING FOR TARGET..."
                    state_color = (0, 180, 255)
                else:
                    state_text = "HOLDING POSITION"
                    state_color = (200, 200, 200)

            # HUD表示
            overlay = view.copy()
            cv2.rectangle(overlay, (12, 12), (460, 140), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.72, view, 0.28, 0, view)
            cv2.putText(view, f"Mode 2: VLM Target Director ({detector.backend.upper()})", (24, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 255), 2)
            cv2.putText(view, f"Target Prompt : '{target_prompt}'", (24, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
            cv2.putText(view, f"Status        : {state_text}", (24, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.56, state_color, 2)
            cv2.putText(view, f"Wyze Command  : Pan {current_pan:+6.1f} / Tilt {current_tilt:+5.1f} deg", (24, 126), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 220, 255), 1)

            if args.record:
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(args.record, fourcc, 15.0, (w, h))
                writer.write(view)

            cv2.imshow("Wyze Cam ROS2 - Mode 2: VLM Target Director", view)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif ord("1") <= key <= ord("6"):
                idx = key - ord("1")
                target_prompt = PRESET_TARGETS[idx]
                print(f"[Target Director] Switched target -> '{target_prompt}'")
            elif key == ord("n"):
                new_t = input("Enter new target prompt (e.g. 'coffee mug', 'glasses'): ").strip()
                if new_t:
                    target_prompt = new_t
    finally:
        if writer is not None:
            writer.release()
        stream.release()
        bus.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
