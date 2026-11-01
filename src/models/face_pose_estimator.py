"""
3D Face Pose Estimator (Yaw / Pitch / Roll) from Static Camera (ATOM Cam 2 / Webcam).
Uses MediaPipe Tasks `FaceLandmarker` (478 3D landmarks + 4x4 Facial Transformation Matrix)
with automatic model download to `/tmp/face_landmarker.task` (avoiding non-ASCII path issues in C++ bindings).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import urllib.request
import shutil
import cv2
import numpy as np


@dataclass
class FacePoseResult:
    yaw_deg: float                  # 左右の顔の向き (-75〜+75 deg, 右向きが正)
    pitch_deg: float                # 上下の顔の向き (-60〜+60 deg, 上向きが正)
    roll_deg: float                 # 首の傾き (-45〜+45 deg)
    bbox: Tuple[int, int, int, int] # (x, y, w, h)
    features: np.ndarray            # 12次元特徴ベクトル (FaceToPTZNet 学習・推論入力用)
    landmarks_2d: np.ndarray        # 描画用主要6点 (6, 2): 鼻先, 顎, 左目, 右目, 口左, 口右


FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
ASCII_MODEL_PATH = Path("/tmp/pai_face_landmarker.task")

# MediaPipe FaceMesh の主要6点インデックス (鼻先, 顎, 左目外端, 右目外端, 口左端, 口右端)
MP_KEY_INDICES = [1, 152, 33, 263, 61, 291]


class FacePoseEstimator:
    """静止カメラ画像から顔の3D角度(Yaw, Pitch, Roll)と12次元特徴量をリアルタイム推定"""

    def __init__(self):
        self.landmarker = None
        self.face_cascade = None
        self._init_mediapipe_landmarker()

    def _init_mediapipe_landmarker(self):
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision

            if not ASCII_MODEL_PATH.exists() or ASCII_MODEL_PATH.stat().st_size < 100_000:
                print(f"[FacePoseEstimator] Downloading MediaPipe FaceLandmarker model to {ASCII_MODEL_PATH}...")
                urllib.request.urlretrieve(FACE_LANDMARKER_URL, str(ASCII_MODEL_PATH))

            base_options = mp_python.BaseOptions(model_asset_path=str(ASCII_MODEL_PATH))
            options = mp_vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=mp_vision.RunningMode.IMAGE,
                num_faces=1,
                output_facial_transformation_matrixes=True,
                min_face_detection_confidence=0.45,
                min_face_presence_confidence=0.45,
            )
            self.landmarker = mp_vision.FaceLandmarker.create_from_options(options)
            print("[FacePoseEstimator] MediaPipe 3D FaceLandmarker initialized successfully!")
            return
        except Exception as e:
            print(f"[FacePoseEstimator] MediaPipe Tasks init failed ({e}), using OpenCV Cascade fallback.")

        # OpenCV Cascade フォールバック（日本語パス回避のため /tmp にコピーしてロード）
        src_xml = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        tmp_xml = Path("/tmp/haarcascade_frontalface_default.xml")
        if src_xml.exists() and not tmp_xml.exists():
            shutil.copyfile(src_xml, tmp_xml)
        self.face_cascade = cv2.CascadeClassifier(str(tmp_xml))

    def estimate(self, frame_bgr: np.ndarray) -> Optional[FacePoseResult]:
        h, w = frame_bgr.shape[:2]
        if self.landmarker is not None:
            return self._estimate_mp_tasks(frame_bgr, w, h)
        return self._estimate_opencv(frame_bgr, w, h)

    def _estimate_mp_tasks(
        self, frame_bgr: np.ndarray, w: int, h: int
    ) -> Optional[FacePoseResult]:
        import mediapipe as mp

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect(mp_image)

        if not result.face_landmarks:
            return None

        lm = result.face_landmarks[0]
        pts_2d = np.array(
            [[lm[idx].x * w, lm[idx].y * h] for idx in MP_KEY_INDICES],
            dtype=np.float64,
        )

        all_x = [p.x * w for p in lm]
        all_y = [p.y * h for p in lm]
        x0, y0 = int(max(0, min(all_x))), int(max(0, min(all_y)))
        x1, y1 = int(min(w, max(all_x))), int(min(h, max(all_y)))
        bbox = (x0, y0, max(1, x1 - x0), max(1, y1 - y0))

        # 1. MediaPipe 4x4 変換行列から3Dオイラー角 (Yaw, Pitch, Roll) を算出
        nose = pts_2d[0]
        chin = pts_2d[1]
        left_eye = pts_2d[2]
        right_eye = pts_2d[3]
        eye_mid = 0.5 * (left_eye + right_eye)
        eye_dist = max(1.0, float(np.linalg.norm(right_eye - left_eye)))
        face_h = max(1.0, float(np.linalg.norm(chin - eye_mid)))

        # ランドマーク非対称性による頑健な角度推定（正面=0, 画面右向き=正, 上向き=正）
        yaw_geom = float(np.clip(((nose[0] - eye_mid[0]) / (0.5 * eye_dist)) * 55.0, -75.0, 75.0))
        pitch_geom = float(np.clip((0.36 - (nose[1] - eye_mid[1]) / face_h) * 135.0, -60.0, 60.0))
        roll_deg = float(
            np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))
        )

        if result.facial_transformation_matrixes:
            mat = np.array(result.facial_transformation_matrixes[0])[:3, :3]
            sy = float(np.sqrt(mat[0, 0] ** 2 + mat[1, 0] ** 2))
            if sy > 1e-6:
                pitch_mat = float(np.degrees(np.arctan2(mat[2, 1], mat[2, 2])))
                yaw_mat = float(np.degrees(np.arctan2(-mat[2, 0], sy)))
                yaw_deg = 0.5 * yaw_geom + 0.5 * yaw_mat
                pitch_deg = 0.5 * pitch_geom + 0.5 * pitch_mat
            else:
                yaw_deg, pitch_deg = yaw_geom, pitch_geom
        else:
            yaw_deg, pitch_deg = yaw_geom, pitch_geom

        features = self._build_feature_vector(yaw_deg, pitch_deg, roll_deg, bbox, pts_2d, w, h)
        return FacePoseResult(
            yaw_deg=yaw_deg,
            pitch_deg=pitch_deg,
            roll_deg=roll_deg,
            bbox=bbox,
            features=features,
            landmarks_2d=pts_2d,
        )

    def _estimate_opencv(
        self, frame_bgr: np.ndarray, w: int, h: int
    ) -> Optional[FacePoseResult]:
        if self.face_cascade is None or self.face_cascade.empty():
            return None
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5)
        if len(faces) == 0:
            return None

        faces = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)
        x, y, fw, fh = [int(v) for v in faces[0]]
        bbox = (x, y, fw, fh)

        cx = (x + 0.5 * fw - 0.5 * w) / (0.5 * w)
        cy = (y + 0.5 * fh - 0.5 * h) / (0.5 * h)
        yaw_deg = float(np.clip(cx * 50.0, -70.0, 70.0))
        pitch_deg = float(np.clip(-cy * 35.0, -50.0, 50.0))
        roll_deg = 0.0

        pts_2d = np.array(
            [
                [x + 0.5 * fw, y + 0.55 * fh],
                [x + 0.5 * fw, y + 0.95 * fh],
                [x + 0.3 * fw, y + 0.38 * fh],
                [x + 0.7 * fw, y + 0.38 * fh],
                [x + 0.35 * fw, y + 0.75 * fh],
                [x + 0.65 * fw, y + 0.75 * fh],
            ],
            dtype=np.float64,
        )
        features = self._build_feature_vector(yaw_deg, pitch_deg, roll_deg, bbox, pts_2d, w, h)
        return FacePoseResult(
            yaw_deg=yaw_deg,
            pitch_deg=pitch_deg,
            roll_deg=roll_deg,
            bbox=bbox,
            features=features,
            landmarks_2d=pts_2d,
        )

    @staticmethod
    def _build_feature_vector(
        yaw_deg: float,
        pitch_deg: float,
        roll_deg: float,
        bbox: Tuple[int, int, int, int],
        pts_2d: np.ndarray,
        w: int,
        h: int,
    ) -> np.ndarray:
        x, y, fw, fh = bbox
        cx_norm = (x + 0.5 * fw) / float(w) - 0.5
        cy_norm = (y + 0.5 * fh) / float(h) - 0.5
        scale_norm = float(fw) / float(w)
        nose_x = pts_2d[0, 0] / float(w) - 0.5
        nose_y = pts_2d[0, 1] / float(h) - 0.5
        leye_x = pts_2d[2, 0] / float(w) - 0.5
        leye_y = pts_2d[2, 1] / float(h) - 0.5
        reye_x = pts_2d[3, 0] / float(w) - 0.5
        reye_y = pts_2d[3, 1] / float(h) - 0.5

        return np.array(
            [
                yaw_deg / 60.0,
                pitch_deg / 45.0,
                roll_deg / 45.0,
                cx_norm,
                cy_norm,
                scale_norm,
                nose_x,
                nose_y,
                leye_x,
                leye_y,
                reye_x,
                reye_y,
            ],
            dtype=np.float32,
        )
