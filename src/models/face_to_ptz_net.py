"""
PyTorch Neural Network (`FaceToPTZNet`) that predicts target Wyze Cam (pan_deg, tilt_deg)
from Static Camera (ATOM Cam 2) 12-D facial pose & landmark features.
"""

from pathlib import Path
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn


class FaceToPTZNet(nn.Module):
    """
    静止カメラ（ATOM Cam 2）で検出した顔姿勢特徴量 (12次元) から、
    別位置に置かれた Wyze Cam Pan v3 の目標 (Pan, Tilt) 角度を推論する残差MLPモデル。
    ベースラインの線形ミラーリング項 + 非線形補正ネットワークにより、少数データでも高精度に収束する。
    """

    def __init__(self, in_features: int = 12, hidden_dim: int = 64):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )
        # 初期状態でも妥当な角度ミラーリングになるよう最終層を小さく初期化
        nn.init.uniform_(self.backbone[-1].weight, -0.05, 0.05)
        nn.init.zeros_(self.backbone[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 12) where x[:, 0] = yaw_deg / 60.0, x[:, 1] = pitch_deg / 45.0
        returns: (B, 2) -> [pan_deg, tilt_deg]
        """
        base_pan = x[:, 0:1] * 60.0
        base_tilt = x[:, 1:2] * 45.0
        base = torch.cat([base_pan, base_tilt], dim=-1)
        residual = self.backbone(x) * 30.0
        return base + residual

    @torch.no_grad()
    def predict_angles(self, features_12d: np.ndarray) -> Tuple[float, float]:
        self.eval()
        tensor_x = torch.from_numpy(features_12d).float().unsqueeze(0)
        out = self.forward(tensor_x)[0].cpu().numpy()
        return float(out[0]), float(out[1])


def load_or_create_model(checkpoint_path: str | Path | None = None) -> FaceToPTZNet:
    model = FaceToPTZNet()
    if checkpoint_path and Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)
        print(f"[FaceToPTZNet] Loaded trained weights from {checkpoint_path}")
    else:
        print("[FaceToPTZNet] Initialized with default residual mirroring weights")
    model.eval()
    return model
