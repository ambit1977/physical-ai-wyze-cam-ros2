#!/usr/bin/env python3
"""
【モード1】静止カメラ（ATOM Cam 2 / Mac内蔵カメラ）で撮影した自分の顔角度に合わせて
ROS2 経由で Wyze Cam Pan v3 を同じ角度に動かすノード（高速VIDEOトラッキング版）。
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
    target_pan: float,
    target_tilt: float,
    fps: float,
    ros_connected: bool,
    dry_run: bool,
):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (12, 12), (430, 145), (20, 20, 20), -1)

    mode_str = "ROS2 Bridge" if ros_connected else "Direct Bridge Proxy"
    if dry_run:
        mode_str += " (DRY-RUN)"

    cv2.putText(frame, f"Mode 1: Face Pose -> Wyze PTZ ({fps:4.1f} FPS)", (24, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 255), 2)
    cv2.putText(frame, f"Face Yaw/Pitch : {yaw:+6.1f} deg / {pitch:+6.1f} deg", (24, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 2)
    cv2.putText(frame, f"Wyze Target PTZ: Pan {target_pan:+6.1f} / Tilt {target_tilt:+5.1f}", (24, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (50, 255, 100), 2)
    cv2.putText(frame, f"Link Status    : {mode_str}  [c: switch cam]", (24, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 220, 255), 1)

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
    parser.add_argument("--cam", type=str, default=Config.STATIC_CAM_SOURCE, help="Camera source ('0' for Mac webcam, or ATOM Cam 2 RTSP)")
    parser.add_argument("--model", type=str, default="data/face_to_ptz.pt", help="Path to trained FaceToPTZNet weights")
    parser.add_argument("--gain-pan", type=float, default=1.2, help="Pan angle multiplier")
    parser.add_argument("--gain-tilt", type=float, default=1.0, help="Tilt angle multiplier")
    parser.add_argument("--ema", type=float, default=0.50, help="EMA smoothing factor (0..1)")
    parser.add_argument("--dry-run", action="store_true", help="Run without sending HTTP requests to Wyze Cam")
    parser.add_argument("--record", type=str, default="", help="Optional MP4 path to record demonstration video")
    args = parser.parse_args()

    current_cam = args.cam
    stream = CameraStream(current_cam)
    estimator = FacePoseEstimator()
    net = load_or_create_model(args.model)
    bus = ROS2WyzeBus(dry_run=args.dry_run)

    writer = None
    smooth_pan = 0.0
    smooth_tilt = 0.0
    pose = None
    fps = 30.0
    last_t = time.perf_counter()

    print("=== Mode 1: Face Angle Mirroring Started ===")
    print("  Keys: [q] Quit | [r] Reset center | [c] Toggle Mac Webcam (0) <-> ATOM Cam 2 RTSP")
    try:
        while True:
            frame, is_new = stream.read_new()
            if frame is None:
                time.sleep(0.005)
                continue

            if not is_new:
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                continue

            now_t = time.perf_counter()
            dt = max(1e-4, now_t - last_t)
            last_t = now_t
            fps = 0.85 * fps + 0.15 * (1.0 / dt)

            view = cv2.flip(frame, 1)
            pose = estimator.estimate(view)

            if pose is not None:
                x, y, bw, bh = pose.bbox
                cv2.rectangle(view, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
                for pt in pose.landmarks_2d:
                    cv2.circle(view, (int(pt[0]), int(pt[1])), 3, (0, 255, 255), -1)

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
                    smooth_pan,
                    smooth_tilt,
                    fps,
                    bus.connected,
                    args.dry_run,
                )
            else:
                draw_hud(view, 0.0, 0.0, smooth_pan, smooth_tilt, fps, bus.connected, args.dry_run)
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
            elif key == ord("c"):
                # 'c' キーで Mac内蔵カメラ (0) と ATOM Cam 2 RTSP をワンタッチ切替
                stream.release()
                current_cam = "0" if current_cam != "0" else Config.STATIC_CAM_SOURCE
                print(f"[Camera] Switching camera source -> {current_cam}")
                stream = CameraStream(current_cam)
    finally:
        if writer is not None:
            writer.release()
        stream.release()
        bus.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
