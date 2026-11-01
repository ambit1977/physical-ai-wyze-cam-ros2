"""
Quest HMD Teleop Demonstration Recorder & Imitation Learning (Behavior Cloning) Policy
for Wyze Cam Pan v3 Search-and-Center Missions.

Records human search-and-center episodes performed via Meta Quest HMD:
  State x_t:
    - target_onehot (4 dims: cup, plastic bottle, cell phone, milk carton)
    - visible_flag  (1 dim : 1.0 if target detected in view, 0.0 if searching)
    - cx_norm, cy_norm, bbox_area_norm (3 dims: visual target coordinates or 0 when unseen)
    - coarse_rgb_grid (4x4x3 = 48 dims: downsampled visual context of current camera view)
    - current_pan_norm, current_tilt_norm (2 dims: current proprioceptive PTZ state)
    - prev_delta_pan, prev_delta_tilt (2 dims: previous velocity memory so search doesn't get stuck)
    Total input dim = 60

  Action u_t:
    - (delta_pan_deg, delta_tilt_deg): Human HMD command step over the next 0.4s window.
"""

from __future__ import annotations

from pathlib import Path
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


DEMO_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "quest_demos"
DEMO_FILE = DEMO_DIR / "quest_search_demos.npz"
MODEL_FILE = DEMO_DIR / "bc_search_policy.pt"


def extract_state_vector(
    frame_bgr: np.ndarray,
    target_idx: int,
    num_targets: int,
    det_center_norm: Optional[Tuple[float, float]],
    det_bbox: Optional[Tuple[int, int, int, int]],
    current_pan: float,
    current_tilt: float,
    prev_dpan: float,
    prev_dtilt: float,
) -> np.ndarray:
    """60次元の観測・自己受容特徴ベクトルを構築"""
    onehot = np.zeros(num_targets, dtype=np.float32)
    onehot[target_idx % num_targets] = 1.0

    h, w = frame_bgr.shape[:2]
    if det_center_norm is not None and det_bbox is not None:
        vis = 1.0
        cx, cy = float(det_center_norm[0]), float(det_center_norm[1])
        _, _, bw, bh = det_bbox
        area = float(bw * bh) / float(max(1, w * h))
    else:
        vis = 0.0
        cx, cy, area = 0.0, 0.0, 0.0

    det_feat = np.array([vis, cx, cy, area], dtype=np.float32)

    # 4x4 RGB Visual Context (48 dims)
    tiny = cv2.resize(frame_bgr, (4, 4), interpolation=cv2.INTER_AREA)
    tiny_rgb = cv2.cvtColor(tiny, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    vis_context = tiny_rgb.flatten()

    proprio = np.array(
        [
            current_pan / 170.0,
            current_tilt / 90.0,
            np.clip(prev_dpan / 30.0, -1.0, 1.0),
            np.clip(prev_dtilt / 20.0, -1.0, 1.0),
        ],
        dtype=np.float32,
    )

    return np.concatenate([onehot, det_feat, vis_context, proprio], axis=0)


class BCSearchNet(nn.Module):
    def __init__(self, in_dim: int = 60, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
            nn.Tanh(),  # Outputs in [-1, 1], scaled by [25.0 deg, 15.0 deg]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        scale = torch.tensor([25.0, 15.0], device=x.device, dtype=x.dtype)
        return out * scale


class QuestDemoBCManager:
    """Questデモ記録・保存・模倣学習(Behavior Cloning)学習・推論を一括管理"""

    def __init__(self, num_targets: int = 4):
        self.num_targets = num_targets
        DEMO_DIR.mkdir(parents=True, exist_ok=True)
        self.states: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.recording: bool = False
        self.episodes_count: int = 0
        self.last_record_ts: float = 0.0
        self.prev_pan_sample: Optional[float] = None
        self.prev_tilt_sample: Optional[float] = None
        self.prev_dpan: float = 0.0
        self.prev_dtilt: float = 0.0

        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.policy: Optional[BCSearchNet] = None
        self.policy_ready: bool = False

        self._load_existing_dataset()
        self._load_existing_policy()

    def _load_existing_dataset(self):
        if DEMO_FILE.exists():
            try:
                data = np.load(DEMO_FILE)
                self.states = list(data["states"])
                self.actions = list(data["actions"])
                self.episodes_count = int(data.get("episodes", len(self.states) // 25))
                print(
                    f"[QuestDemoBC] Loaded {len(self.states)} demo steps ({self.episodes_count} episodes) from {DEMO_FILE.name}"
                )
            except Exception as e:
                print(f"[QuestDemoBC] Could not load dataset: {e}")

    def _load_existing_policy(self):
        if MODEL_FILE.exists():
            try:
                net = BCSearchNet().to(self.device)
                net.load_state_dict(torch.load(MODEL_FILE, map_location=self.device))
                net.eval()
                self.policy = net
                self.policy_ready = True
                print(f"[QuestDemoBC] Loaded trained BC policy from {MODEL_FILE.name}")
            except Exception as e:
                print(f"[QuestDemoBC] Could not load policy: {e}")

    def toggle_recording(self, current_pan: float, current_tilt: float) -> bool:
        self.recording = not self.recording
        if self.recording:
            self.prev_pan_sample = current_pan
            self.prev_tilt_sample = current_tilt
            self.prev_dpan = 0.0
            self.prev_dtilt = 0.0
            self.last_record_ts = time.time()
            print("[QuestDemoBC] 🔴 STARTED recording Quest demonstration episode!")
        else:
            self.episodes_count += 1
            self.save_dataset()
            print(
                f"[QuestDemoBC] ⏹ STOPPED recording. Total episodes: {self.episodes_count} ({len(self.states)} steps)."
            )
        return self.recording

    def record_step_if_active(
        self,
        frame_bgr: np.ndarray,
        target_idx: int,
        det_center_norm: Optional[Tuple[float, float]],
        det_bbox: Optional[Tuple[int, int, int, int]],
        cmd_pan: float,
        cmd_tilt: float,
    ):
        if not self.recording:
            return
        now = time.time()
        if now - self.last_record_ts < 0.25:  # 4Hz sampling matching PTZ settle timescale
            return
        self.last_record_ts = now

        if self.prev_pan_sample is None or self.prev_tilt_sample is None:
            self.prev_pan_sample = cmd_pan
            self.prev_tilt_sample = cmd_tilt
            return

        dpan = float(np.clip(cmd_pan - self.prev_pan_sample, -25.0, 25.0))
        dtilt = float(np.clip(cmd_tilt - self.prev_tilt_sample, -15.0, 15.0))

        s_vec = extract_state_vector(
            frame_bgr=frame_bgr,
            target_idx=target_idx,
            num_targets=self.num_targets,
            det_center_norm=det_center_norm,
            det_bbox=det_bbox,
            current_pan=self.prev_pan_sample,
            current_tilt=self.prev_tilt_sample,
            prev_dpan=self.prev_dpan,
            prev_dtilt=self.prev_dtilt,
        )
        a_vec = np.array([dpan, dtilt], dtype=np.float32)

        self.states.append(s_vec)
        self.actions.append(a_vec)

        self.prev_pan_sample = cmd_pan
        self.prev_tilt_sample = cmd_tilt
        self.prev_dpan = dpan
        self.prev_dtilt = dtilt

    def save_dataset(self):
        if not self.states:
            return
        np.savez_compressed(
            DEMO_FILE,
            states=np.stack(self.states, axis=0),
            actions=np.stack(self.actions, axis=0),
            episodes=self.episodes_count,
        )

    def train_policy(self, epochs: int = 80, lr: float = 3e-3) -> Tuple[bool, str]:
        if len(self.states) < 8:
            msg = f"Need at least 8 steps to train (currently {len(self.states)} steps)."
            print(f"[QuestDemoBC] {msg}")
            return False, msg

        X = torch.tensor(np.stack(self.states, axis=0), dtype=torch.float32, device=self.device)
        Y = torch.tensor(np.stack(self.actions, axis=0), dtype=torch.float32, device=self.device)

        net = BCSearchNet(in_dim=X.shape[1]).to(self.device)
        net.train()
        optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
        loss_fn = nn.SmoothL1Loss()

        final_loss = 0.0
        for _ in range(epochs):
            optimizer.zero_grad()
            pred = net(X)
            loss = loss_fn(pred, Y)
            loss.backward()
            optimizer.step()
            final_loss = float(loss.item())

        net.eval()
        torch.save(net.state_dict(), MODEL_FILE)
        self.policy = net
        self.policy_ready = True
        msg = f"Trained BC Policy on {len(self.states)} steps ({self.episodes_count} eps), Loss={final_loss:.3f}"
        print(f"[QuestDemoBC] ✅ {msg}")
        return True, msg

    def predict_delta(
        self,
        frame_bgr: np.ndarray,
        target_idx: int,
        det_center_norm: Optional[Tuple[float, float]],
        det_bbox: Optional[Tuple[int, int, int, int]],
        current_pan: float,
        current_tilt: float,
    ) -> Optional[Tuple[float, float]]:
        if not self.policy_ready or self.policy is None:
            return None
        s_vec = extract_state_vector(
            frame_bgr=frame_bgr,
            target_idx=target_idx,
            num_targets=self.num_targets,
            det_center_norm=det_center_norm,
            det_bbox=det_bbox,
            current_pan=current_pan,
            current_tilt=current_tilt,
            prev_dpan=self.prev_dpan,
            prev_dtilt=self.prev_dtilt,
        )
        with torch.no_grad():
            x = torch.tensor(s_vec[None, :], dtype=torch.float32, device=self.device)
            out = self.policy(x)[0].cpu().numpy()
        dpan, dtilt = float(out[0]), float(out[1])
        self.prev_dpan = dpan
        self.prev_dtilt = dtilt
        return dpan, dtilt
