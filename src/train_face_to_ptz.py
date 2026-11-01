#!/usr/bin/env python3
"""
Train `FaceToPTZNet`:
静止カメラ（ATOM Cam 2）での顔姿勢特徴量 (12次元) と、
Wyze Cam Pan v3 が向くべき目標角度 (Pan, Tilt) の対応を学習するスクリプト。

使い方:
  # 1. 合成キャリブレーションデータ + 非線形カメラ配置補正を含めて即座にモデル学習 (約3秒)
  python -m src.train_face_to_ptz --out data/face_to_ptz.pt

  # 2. 実機カメラの前で5方向（正面・左・右・上・下）を向いてキャリブレーション収集＆学習
  python -m src.train_face_to_ptz --interactive --cam 0 --out data/face_to_ptz.pt
"""

import argparse
from pathlib import Path
import time
import cv2
import numpy as np
import torch
import torch.nn as nn

from src.camera_stream import CameraStream
from src.models.face_pose_estimator import FacePoseEstimator
from src.models.face_to_ptz_net import FaceToPTZNet


CALIBRATION_TARGETS = [
    ("Center (正面を向いてください)", 0.0, 0.0),
    ("Left -45 deg (左を向いてください)", -45.0, 0.0),
    ("Right +45 deg (右を向いてください)", 45.0, 0.0),
    ("Up +25 deg (上を向いてください)", 0.0, 25.0),
    ("Down -20 deg (下を向いてください)", 0.0, -20.0),
]


def generate_synthetic_calibration_dataset(
    num_samples: int = 800,
    camera_offset_yaw: float = 5.0,
    camera_offset_pitch: float = -3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    ATOM Cam 2 と Wyze Cam Pan v3 の設置位置オフセット・視野角非線形性を模擬した
    学習データセット (X: [N, 12], Y: [N, 2]) を生成する。
    """
    rng = np.random.default_rng(42)
    yaws = rng.uniform(-65.0, 65.0, size=num_samples)
    pitches = rng.uniform(-40.0, 40.0, size=num_samples)
    rolls = rng.uniform(-15.0, 15.0, size=num_samples)
    cx = rng.uniform(-0.35, 0.35, size=num_samples)
    cy = rng.uniform(-0.30, 0.30, size=num_samples)
    scale = rng.uniform(0.18, 0.42, size=num_samples)

    X = np.zeros((num_samples, 12), dtype=np.float32)
    Y = np.zeros((num_samples, 2), dtype=np.float32)

    for i in range(num_samples):
        yaw = yaws[i]
        pitch = pitches[i]
        roll = rolls[i]
        nose_x = cx[i] - (yaw / 60.0) * 0.08
        nose_y = cy[i] - (pitch / 45.0) * 0.06
        X[i] = np.array(
            [
                yaw / 60.0,
                pitch / 45.0,
                roll / 45.0,
                cx[i],
                cy[i],
                scale[i],
                nose_x,
                nose_y,
                cx[i] - 0.07,
                cy[i] - 0.04,
                cx[i] + 0.07,
                cy[i] - 0.04,
            ],
            dtype=np.float32,
        )
        # 目標Wyze角度: 顔のYaw/Pitch角度 + 画面内の顔位置による視差補正 + カメラ設置オフセット
        target_pan = yaw * 1.15 + cx[i] * 18.0 + camera_offset_yaw
        target_tilt = pitch * 1.10 - cy[i] * 12.0 + camera_offset_pitch
        Y[i, 0] = np.clip(target_pan, -170.0, 170.0)
        Y[i, 1] = np.clip(target_tilt, -40.0, 80.0)

    return X, Y


def collect_interactive_dataset(cam_source: str) -> tuple[np.ndarray, np.ndarray]:
    stream = CameraStream(cam_source)
    estimator = FacePoseEstimator()
    xs, ys = [], []

    print("=== 5点インタラクティブ・キャリブレーション開始 ===")
    for label, target_pan, target_tilt in CALIBRATION_TARGETS:
        print(f"Next: {label} -> SPACEキーで30フレーム記録")
        recording = False
        collected = 0
        while collected < 30:
            frame = stream.read()
            if frame is None:
                time.sleep(0.02)
                continue
            disp = frame.copy()
            pose = estimator.estimate(frame)
            status = "RECORDING..." if recording else "Press SPACE to record"
            cv2.putText(
                disp,
                f"Target: {label} ({target_pan:+.0f}, {target_tilt:+.0f})",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                disp,
                f"{status} [{collected}/30]",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0) if recording else (255, 200, 0),
                2,
            )
            if pose is not None and recording:
                xs.append(pose.features)
                ys.append([target_pan, target_tilt])
                collected += 1
            cv2.imshow("Calibration", disp)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                recording = True
            elif key == ord("q"):
                break

    stream.release()
    cv2.destroyAllWindows()
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def train_model(X: np.ndarray, Y: np.ndarray, epochs: int = 250, lr: float = 3e-3) -> FaceToPTZNet:
    model = FaceToPTZNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.SmoothL1Loss()

    tensor_x = torch.from_numpy(X).float()
    tensor_y = torch.from_numpy(Y).float()

    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        pred = model(tensor_x)
        loss = criterion(pred, tensor_y)
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch % 50 == 0 or epoch == epochs:
            mae = torch.mean(torch.abs(pred - tensor_y)).item()
            print(f"[Epoch {epoch:03d}/{epochs:03d}] Loss={loss.item():.4f} | Angle MAE={mae:.2f} deg")

    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Train FaceToPTZNet calibration model")
    parser.add_argument("--interactive", action="store_true", help="Collect live 5-point calibration from camera")
    parser.add_argument("--cam", type=str, default="0", help="Static camera source (ATOM Cam 2 RTSP or USB index)")
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--out", type=str, default="data/face_to_ptz.pt")
    args = parser.parse_args()

    if args.interactive:
        X, Y = collect_interactive_dataset(args.cam)
    else:
        X, Y = generate_synthetic_calibration_dataset()

    model = train_model(X, Y, epochs=args.epochs)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict()}, out_path)
    print(f"Saved trained FaceToPTZNet model -> {out_path}")


if __name__ == "__main__":
    main()
