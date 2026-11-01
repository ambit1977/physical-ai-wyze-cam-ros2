"""
Cross-Embodiment Retargeter: Multi-Axis Robot Arm (6-DoF / 7-DoF) -> Camera Pan-Tilt (PT) & Gaze Vector.

多軸ロボットアーム（LIBERO / OpenVLA / Octo / Quest 6DoF 操作などの 7次元アクション
`[dx, dy, dz, droll, dpitch, dyaw, gripper]` または 3D手先位置 `p_ee = [x, y, z]`）を、
Wyze Cam Pan v3 の 2軸 (Pan, Tilt) 回転角および 3次元視線ベクトル (Gaze Line-of-Sight) に変換する数理モジュール。

数理モデル (Spherical Gaze Projection & Differential Retargeting):
  1. カメラ光学中心を原点 O とし、カメラ前方を +Z (基準作業距離 Z_0 = 0.65m)、右方を +X、上方を +Y とする。
  2. 多軸アームの手先目標位置 p_ee = [x, y, z]^T および手首回転 [roll, pitch, yaw]^T に対し、
     カメラから手先作用点へ向かう 3D 視線単位ベクトル (Gaze Vector) v_gaze を次式で定義する:
       v_gaze = p_ee / ||p_ee||_2 = [g_x, g_y, g_z]^T
  3. 球面座標ヤコビアン投影により、手先変位 (dx, dy, dz) と手首回転 (dyaw, dpitch) を
     カメラの Pan / Tilt 角度変化 (d_pan, d_tilt) へ変換する:
       d_pan  = atan2(dx, Z_0 + dz) * (180 / pi) + w_rot * dyaw_deg
       d_tilt = atan2(dy, sqrt(dx^2 + (Z_0 + dz)^2)) * (180 / pi) + w_rot * dpitch_deg
  4. 奥行軸 (dz) とグリッパ軸 (gripper) は「視線の注視度 (Foveation / Approach Damping)」に変換し、
     対象物へ手先が近づく (dz > 0, gripper close) ほどサーボのステップゲインを絞って精密に中心ロックする。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import math
import numpy as np


@dataclass
class GazePTZCommand:
    """多軸アーム動作から変換されたカメラPT軸・視線ベクトル出力"""

    pan_deg: float                     # 目標 Pan 角度 [deg] (-170 ~ +170)
    tilt_deg: float                    # 目標 Tilt 角度 [deg] (-45 ~ +90)
    delta_pan_deg: float               # 今回のステップの Pan 増分 [deg]
    delta_tilt_deg: float              # 今回のステップの Tilt 増分 [deg]
    gaze_vector: Tuple[float, float, float]  # 3D単位視線ベクトル (gx, gy, gz) in Camera/World Frame
    virtual_ee_pos: Tuple[float, float, float]  # 対応する仮想7軸アーム手先3D座標 (x, y, z) [m]
    foveation_zoom: float              # 注視ズーム倍率 (1.0 ~ 2.0: アームリーチ dz / gripper に対応)
    phase: str                         # "LOCKED" | "SERVO_CENTERING" | "SEARCHING_WORKSPACE"


class ArmToPTZGazeRetargeter:
    """
    7軸ロボットアーム動作 `[dx, dy, dz, droll, dpitch, dyaw, gripper]` または
    画像上の2Dターゲット位置 / 3D探索軌道を、カメラのPT軸と視線ベクトルへ変換するクラス。
    """

    # 人間が机上や周囲を見回す際の「3D作業空間内の仮想アーム探索ウェイポイント (x[m], y[m], z[m])」
    # カメラ前方 z = 0.65m を基準に、正面 → 左手前 → 左奥 → 左上 → 中央上 → 右上 → 右奥 → 右手前 → 手前机上 を巡回
    WORKSPACE_SEARCH_WAYPOINTS_3D: List[Tuple[float, float, float, str]] = [
        (0.00,  0.00, 0.65, "Center (正面)"),
        (-0.38, -0.06, 0.65, "Left-Front (左前・机上)"),
        (-0.78, -0.06, 0.65, "Left-Wide (左側)"),
        (-0.38, +0.16, 0.65, "Left-Up (左上)"),
        (0.00,  +0.18, 0.65, "Center-Up (正面やや上)"),
        (+0.38, +0.16, 0.65, "Right-Up (右上)"),
        (+0.78, -0.06, 0.65, "Right-Wide (右側)"),
        (+0.38, -0.06, 0.65, "Right-Front (右前・机上)"),
        (0.00,  -0.18, 0.65, "Center-Down (手前・机上)"),
    ]

    def __init__(
        self,
        nominal_depth_m: float = 0.65,
        hfov_deg: float = 78.0,
        vfov_deg: float = 46.0,
        servo_settle_sec: float = 0.72,
        search_dwell_sec: float = 1.45,
        max_step_pan_deg: float = 11.0,
        max_step_tilt_deg: float = 7.5,
    ):
        self.nominal_depth_m = nominal_depth_m
        self.hfov_rad = math.radians(hfov_deg)
        self.vfov_rad = math.radians(vfov_deg)
        self.servo_settle_sec = servo_settle_sec
        self.search_dwell_sec = search_dwell_sec
        self.max_step_pan_deg = max_step_pan_deg
        self.max_step_tilt_deg = max_step_tilt_deg

        # 状態変数 (RTSP遅延によるワインドアップ暴走を完全に防ぐ Step-and-Settle 制御用)
        self.last_step_ts: float = 0.0
        self.last_seen_ts: float = 0.0
        self.confirm_streak: int = 0
        self.search_wp_idx: int = 0
        self.search_wp_label: str = "Center (正面)"
        self.virtual_ee_xyz: np.ndarray = np.array([0.0, 0.0, nominal_depth_m], dtype=np.float64)

    @staticmethod
    def _xyz_to_gaze_and_ptz(x: float, y: float, z: float) -> Tuple[Tuple[float, float, float], float, float]:
        """3D作業空間座標 (x, y, z) から単位視線ベクトル v_gaze と球面 (Pan, Tilt) 角度を算出"""
        z_safe = max(0.15, float(z))
        norm = math.sqrt(x * x + y * y + z_safe * z_safe)
        gx, gy, gz = x / norm, y / norm, z_safe / norm
        pan_deg = math.degrees(math.atan2(x, z_safe))
        tilt_deg = math.degrees(math.atan2(y, math.sqrt(x * x + z_safe * z_safe)))
        return (gx, gy, gz), pan_deg, tilt_deg

    @staticmethod
    def ptz_to_gaze_vector(pan_deg: float, tilt_deg: float, depth_m: float = 0.65) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """現在のカメラ (Pan, Tilt) 角度から 3D 視線単位ベクトルと仮想手先座標 (x, y, z) を逆算"""
        pan_rad = math.radians(pan_deg)
        tilt_rad = math.radians(tilt_deg)
        gx = math.cos(tilt_rad) * math.sin(pan_rad)
        gy = math.sin(tilt_rad)
        gz = math.cos(tilt_rad) * math.cos(pan_rad)
        ee_x = depth_m * gx
        ee_y = depth_m * gy
        ee_z = depth_m * gz
        return (gx, gy, gz), (ee_x, ee_y, ee_z)

    def retarget_7dof_arm_action(
        self,
        arm_action_7d: np.ndarray,
        current_pan: float,
        current_tilt: float,
        rot_weight: float = 0.35,
    ) -> GazePTZCommand:
        """
        7軸ロボットアームの行動ベクトル `a = [dx, dy, dz, droll, dpitch, dyaw, gripper]` を
        カメラの (Pan, Tilt) 角度と視線ベクトルへ直接変換する。
        """
        dx, dy, dz, droll, dpitch, dyaw, gripper = [float(v) for v in arm_action_7d[:7]]
        depth = max(0.20, self.nominal_depth_m + dz)

        # 球面ヤコビアンによる並進 (dx, dy, dz) -> 視線角度変換 + 手首回転 (dyaw, dpitch) の合成
        dpan_trans = math.degrees(math.atan2(dx, depth))
        dtilt_trans = math.degrees(math.atan2(dy, math.sqrt(dx * dx + depth * depth)))

        dpan = float(np.clip(dpan_trans + rot_weight * math.degrees(dyaw), -self.max_step_pan_deg, self.max_step_pan_deg))
        dtilt = float(np.clip(dtilt_trans + rot_weight * math.degrees(dpitch), -self.max_step_tilt_deg, self.max_step_tilt_deg))

        new_pan = float(np.clip(current_pan + dpan, -170.0, 170.0))
        new_tilt = float(np.clip(current_tilt + dtilt, -45.0, 90.0))

        gaze_vec, ee_pos = self.ptz_to_gaze_vector(new_pan, new_tilt, depth_m=depth)
        foveation = float(np.clip(1.0 + max(0.0, dz) + 0.5 * max(0.0, gripper), 1.0, 2.0))

        return GazePTZCommand(
            pan_deg=new_pan,
            tilt_deg=new_tilt,
            delta_pan_deg=dpan,
            delta_tilt_deg=dtilt,
            gaze_vector=gaze_vec,
            virtual_ee_pos=ee_pos,
            foveation_zoom=foveation,
            phase="ARM_RETARGET",
        )

    def image_error_to_7dof_arm_action(
        self, cx_norm: float, cy_norm: float, bbox_area_norm: float
    ) -> np.ndarray:
        """
        画像上のターゲット偏差 (cx_norm, cy_norm) を、一度「仮想7軸アームが対象物へ手を伸ばす動作
        `[dx, dy, dz, droll, dpitch, dyaw, gripper]`」として表現する。
        """
        # カメラ座標系での対象物の横ズレ dx, 縦ズレ dy [m]
        dx = 2.0 * cx_norm * self.nominal_depth_m * math.tan(0.5 * self.hfov_rad) * 0.42
        dy = -2.0 * cy_norm * self.nominal_depth_m * math.tan(0.5 * self.vfov_rad) * 0.42
        # 中心に近づくほど手先を前進 (dz > 0) させ、グリッパを閉じる (注視ロック)
        err_norm = math.hypot(cx_norm, cy_norm)
        dz = 0.25 * max(0.0, 0.25 - err_norm)
        dyaw = math.atan2(dx, self.nominal_depth_m)
        dpitch = math.atan2(dy, self.nominal_depth_m)
        gripper = 1.0 if err_norm < 0.085 else 0.0
        return np.array([dx, dy, dz, 0.0, dpitch, dyaw, gripper], dtype=np.float32)

    def step_search_and_center(
        self,
        now: float,
        det_center_norm: Tuple[float, float] | None,
        det_bbox_area_norm: float,
        current_pan: float,
        current_tilt: float,
        bc_delta: Tuple[float, float] | None = None,
    ) -> GazePTZCommand:
        """
        【暴走防止 Step-and-Settle 制御 + 人間らしい探索 & 視線中心化】
        - 視界に対象が入っている場合:
          仮想7軸アームのリーチ動作 `[dx, dy, dz, ...]` を生成し、それをPT軸・視線ベクトルに変換。
          RTSP遅延 (約0.6秒) による積分ワインドアップ暴走を防ぐため、1ステップ最大 ±11° 動いたら
          `servo_settle_sec (0.72秒)` 待機して映像が落ち着くのを待ってから次ステップを計算する。
        - 視界に対象が入っていない場合:
          人間が辺りを見回すように 3D 作業空間ウェイポイント (または Quest 模倣学習 BC ポリシー) に
          視線ベクトルを向け、各ポイントで `search_dwell_sec (1.45秒)` 止まって対象物を探す。
        """
        # 1. ターゲットが検出されている場合
        if det_center_norm is not None:
            self.confirm_streak += 1
            self.last_seen_ts = now
            cx_norm, cy_norm = det_center_norm

            # まだ1フレーム目だけの瞬間ノイズなら現在位置で様子を見る (誤検出による暴走防止)
            if self.confirm_streak < 2:
                gaze_vec, ee_pos = self.ptz_to_gaze_vector(current_pan, current_tilt, self.nominal_depth_m)
                return GazePTZCommand(
                    pan_deg=current_pan,
                    tilt_deg=current_tilt,
                    delta_pan_deg=0.0,
                    delta_tilt_deg=0.0,
                    gaze_vector=gaze_vec,
                    virtual_ee_pos=ee_pos,
                    foveation_zoom=1.2,
                    phase="VERIFYING_TARGET",
                )

            # すでに画面中心 (デッドバンド |cx|<0.08, |cy|<0.10) に入っていればピタッと停止してロック！
            if abs(cx_norm) < 0.08 and abs(cy_norm) < 0.10:
                gaze_vec, ee_pos = self.ptz_to_gaze_vector(current_pan, current_tilt, self.nominal_depth_m + 0.15)
                return GazePTZCommand(
                    pan_deg=current_pan,
                    tilt_deg=current_tilt,
                    delta_pan_deg=0.0,
                    delta_tilt_deg=0.0,
                    gaze_vector=gaze_vec,
                    virtual_ee_pos=ee_pos,
                    foveation_zoom=1.8,
                    phase="LOCKED_CENTER",
                )

            # 前回のPTZコマンド送信から servo_settle_sec (0.72秒) 未満なら、カメラ移動＆RTSP映像更新待ち！
            # ★これが「ぐるっと回ってしまう暴走」を100%防ぐ核心ロジック
            if now - self.last_step_ts < self.servo_settle_sec:
                gaze_vec, ee_pos = self.ptz_to_gaze_vector(current_pan, current_tilt, self.nominal_depth_m)
                return GazePTZCommand(
                    pan_deg=current_pan,
                    tilt_deg=current_tilt,
                    delta_pan_deg=0.0,
                    delta_tilt_deg=0.0,
                    gaze_vector=gaze_vec,
                    virtual_ee_pos=ee_pos,
                    foveation_zoom=1.3,
                    phase="SETTLING_WAIT",
                )

            # 7軸アームのリーチ動作を生成し、PT軸・視線ベクトルへリターゲティング
            self.last_step_ts = now
            arm_action = self.image_error_to_7dof_arm_action(cx_norm, cy_norm, det_bbox_area_norm)
            cmd = self.retarget_7dof_arm_action(arm_action, current_pan, current_tilt)

            # Quest模倣学習(BC)ポリシーが学習済みの場合は25%ブレンドして人間の滑らかな加減速を反映
            if bc_delta is not None:
                bc_dpan, bc_dtilt = bc_delta
                blended_pan = float(np.clip(current_pan + 0.75 * cmd.delta_pan_deg + 0.25 * bc_dpan, -170.0, 170.0))
                blended_tilt = float(np.clip(current_tilt + 0.75 * cmd.delta_tilt_deg + 0.25 * bc_dtilt, -45.0, 90.0))
                gaze_vec, ee_pos = self.ptz_to_gaze_vector(blended_pan, blended_tilt, self.nominal_depth_m)
                return GazePTZCommand(
                    pan_deg=blended_pan,
                    tilt_deg=blended_tilt,
                    delta_pan_deg=blended_pan - current_pan,
                    delta_tilt_deg=blended_tilt - current_tilt,
                    gaze_vector=gaze_vec,
                    virtual_ee_pos=ee_pos,
                    foveation_zoom=cmd.foveation_zoom,
                    phase="SERVO_CENTERING (Arm+BC)",
                )

            cmd.phase = "SERVO_CENTERING (Arm->PTZ)"
            return cmd

        # 2. ターゲットが視界に無い場合 (人間のように辺りを見回して探す)
        self.confirm_streak = 0

        # 見失ってから 1.2秒間はその場で静止して再検出を待つ
        if self.last_seen_ts > 0.0 and (now - self.last_seen_ts < 1.2):
            gaze_vec, ee_pos = self.ptz_to_gaze_vector(current_pan, current_tilt, self.nominal_depth_m)
            return GazePTZCommand(
                pan_deg=current_pan,
                tilt_deg=current_tilt,
                delta_pan_deg=0.0,
                delta_tilt_deg=0.0,
                gaze_vector=gaze_vec,
                virtual_ee_pos=ee_pos,
                foveation_zoom=1.0,
                phase="HOLD_REACQUIRE",
            )

        # 各探索ウェイポイントで search_dwell_sec (1.45秒) しっかり止まって周囲を観察する
        if now - self.last_step_ts < self.search_dwell_sec:
            gaze_vec, ee_pos = self.ptz_to_gaze_vector(current_pan, current_tilt, self.nominal_depth_m)
            return GazePTZCommand(
                pan_deg=current_pan,
                tilt_deg=current_tilt,
                delta_pan_deg=0.0,
                delta_tilt_deg=0.0,
                gaze_vector=gaze_vec,
                virtual_ee_pos=ee_pos,
                foveation_zoom=1.0,
                phase=f"SEARCH_LOOKING [{self.search_wp_label}]",
            )

        # 次の3D探索ウェイポイントへ仮想アーム手先 (wx, wy, wz) を移動し、そこからカメラPT角度と視線ベクトルを算出
        self.last_step_ts = now
        wx, wy, wz, wp_label = self.WORKSPACE_SEARCH_WAYPOINTS_3D[
            self.search_wp_idx % len(self.WORKSPACE_SEARCH_WAYPOINTS_3D)
        ]
        self.search_wp_idx = (self.search_wp_idx + 1) % len(self.WORKSPACE_SEARCH_WAYPOINTS_3D)
        self.search_wp_label = wp_label

        gaze_vec, target_pan, target_tilt = self._xyz_to_gaze_and_ptz(wx, wy, wz)

        # Quest模倣学習(BC)ポリシーがある場合は、人間の探索ステップと3Dウェイポイント探索を融合
        if bc_delta is not None:
            bc_dpan, bc_dtilt = bc_delta
            target_pan = float(np.clip(0.7 * target_pan + 0.3 * (current_pan + bc_dpan), -85.0, 85.0))
            target_tilt = float(np.clip(0.7 * target_tilt + 0.3 * (current_tilt + bc_dtilt), -25.0, 35.0))
            gaze_vec, (wx, wy, wz) = self.ptz_to_gaze_vector(target_pan, target_tilt, self.nominal_depth_m)

        return GazePTZCommand(
            pan_deg=target_pan,
            tilt_deg=target_tilt,
            delta_pan_deg=target_pan - current_pan,
            delta_tilt_deg=target_tilt - current_tilt,
            gaze_vector=gaze_vec,
            virtual_ee_pos=(wx, wy, wz),
            foveation_zoom=1.0,
            phase=f"SEARCH_MOVE -> {wp_label}",
        )
