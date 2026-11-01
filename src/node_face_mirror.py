#!/usr/bin/env python3
"""
【モード1】静止カメラ（ATOM Cam 2 / USBカメラ）で撮影した自分の顔角度に合わせて
ROS2 経由で Wyze Cam Pan v3 を同じ角度に動かすノード。

特徴:
  - MediaPipe 3D Face Mesh (solvePnP) で顔の Yaw / Pitch / Roll を推定
  - PyTorch `FaceToPTZNet` モデルで ATOM Cam 2 座標系から Wyze Cam 目標角度 (Pan, Tilt) へ変換
  - EMA平滑化フィルタ + ROS2 `/wyze/cmd_ptz` & `/cmd_vel` へのリアルタイムPublish
  - 画面上に3D顔角度・目標カメラ角度・仮想PTZレーダーを描画し、`--record` で提出用動画も保存可能
"""

import argparse
import time
import cv2
import numpy as np

from src.config import Config
from src.camera_stream import CameraStream
from src.ros2_bus import ROS2WyzeBus
from src.models.face_pose_estimator import FacePoseEstimator
from src.models.face_to_ptz_net import load_or_create_model


def draw_hud(
    frame: np.ndarray,
    yaw: float,
    pitch: float,
    roll: float,
    target_pan: float,
    target_tilt: float,
    ros_connected: bool,
    dry_run: bool,
):
    h, w = frame.shape[:2]
    # 半透明ステータスパネル
    overlay = frame.copy()
    cv2.rectangle(overlay, (12, 12), (420, 145), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    mode_str = "ROS2 Bridge" if ros_connected else "Direct Bridge Proxy"
    if dry_run:
        mode_str += " (DRY-RUN)"

    cv2.putText(frame, "Mode 1: Face Pose -> Wyze PTZ Mirror", (24, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2)
    cv2.putText(frame, f"Face Yaw/Pitch : {yaw:+6.1f} deg / {pitch:+6.1f} deg", (24, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 2)
    cv2.putText(frame, f"Wyze Target PTZ: Pan {target_pan:+6.1f} / Tilt {target_tilt:+5.1f}", (24, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (50, 255, 100), 2)
    cv2.putText(frame, f"Link Status    : {mode_str}", (24, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 220, 255), 1)

    # 右下にWyze Camの向きを示す仮想PTZレーダーを描画
    radar_cx, radar_cy, radar_r = w - 85, h - 85, 62
    cv2.circle(frame, (radar_cx, radar_cy), radar_r, (40, 40, 40), -1)
    cv2.circle(frame, (radar_cx, radar_cy), radar_r, (200, 200, 200), 2)
    cv2.line(frame, (radar_cx - radar_r, radar_cy), (radar_cx + radar_r, radar_cy), (90, 90, 90), 1)
    cv2.line(frame, (radar_cx, radar_cy - radar_r), (radar_cx, radar_cy + radar_r), (90, 90, 90), 1)

    dot_x = int(radar_cx + np.clip(target_pan / 90.0, -1.0, 1.0) * (radar_r - 8))
    dot_y = int(radar_cy - np.clip(target_tilt / 45.0, -1.0, 1.0) * (radar_r - 8))
    cv2.arrowedLine(frame, (radar_cx, radar_cy), (dot_x, dot_y), (0, 255, 100), 3, tipLength=0.25)
    cv2.putText(frame, "Wyze PTZ", (radar_cx - 38, radar_cy + radar_r + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)


def main():
    parser = argparse.ArgumentParser(description="Mode 1: Face Angle Mirroring to Wyze Cam Pan v3 via ROS2")
    parser.add_argument("--cam", type=str, default=Config.STATIC_CAM_SOURCE, help="ATOM Cam 2 RTSP URL or USB camera index (e.g., 0)")
    parser.add_argument("--model", type=str, default="data/face_to_ptz.pt", help="Path to trained FaceToPTZNet weights")
    parser.add_argument("--gain-pan", type=float, default=1.2, help="Pan angle multiplier")
    parser.add_argument("--gain-tilt", type=float, default=1.0, help="Tilt angle multiplier")
    parser.add_argument("--ema", type=float, default=0.35, help="EMA smoothing factor (0..1)")
    parser.add_argument("--dry-run", action="store_true", help="Run without sending HTTP requests to Wyze Cam")
    parser.add_argument("--record", type=str, default="", help="Optional MP4 path to record demonstration video")
    args = parser.parse_args()

    stream = CameraStream(args.cam)
    estimator = FacePoseEstimator()
    net = load_or_create_model(args.model)
    bus = ROS2WyzeBus(dry_run=args.dry_run)

    writer = None
    smooth_pan = 0.0
    smooth_tilt = 0.0

    print("=== Mode 1: Face Angle Mirroring Started (Press 'q' to quit, 'r' to reset center) ===")
    try:
        while True:
            frame = stream.read()
            if frame is None:
                time.sleep(0.02)
                continue

            # 鏡像表示（自分が向きやすいよう左右反転）
            view = cv2.flip(frame, 1)
            pose = estimator.estimate(view)

            if pose is not None:
                # ランドマークとバウンディングボックス描画
                x, y, bw, bh = pose.bbox
                cv2.rectangle(view, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
                for pt in pose.landmarks_2d:
                    cv2.circle(view, (int(pt[0]), int(pt[1])), 3, (0, 255, 255), -1)

                # PyTorchモデルで目標Pan/Tilt角度を推論
                raw_pan, raw_tilt = net.predict_angles(pose.features)
                raw_pan *= args.gain_pan
                raw_tilt *= args.gain_tilt

                smooth_pan = (1.0 - args.ema) * smooth_pan + args.ema * raw_pan
                smooth_tilt = (1.0 - args.ema) * smooth_tilt + args.ema * raw_tilt

                bus.publish_ptz(smooth_pan, smooth_tilt)
                draw_hud(
                    view,
                    pose.yaw_deg,
                    pose.pitch_deg,
                    pose.roll_deg,
                    smooth_pan,
                    smooth_tilt,
                    bus.connected,
                    args.dry_run,
                )
            else:
                draw_hud(view, 0.0, 0.0, 0.0, smooth_pan, smooth_tilt, bus.connected, args.dry_run)
                cv2.putText(view, "No face detected", (24, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)

            if args.record:
                if writer is None:
                    h, w = view.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(args.record, fourcc, 20.0, (w, h))
                writer.write(view)

            cv2.imshow("Wyze Cam ROS2 - Mode 1: Face Pose Mirror", view)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                smooth_pan, smooth_tilt = 0.0, 0.0
                bus.publish_ptz(0.0, 0.0)
    finally:
        if writer is not None:
            writer.release()
        stream.release()
        bus.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
