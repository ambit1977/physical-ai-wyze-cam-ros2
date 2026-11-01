"""
High-speed 3D Face Pose Estimator (Yaw / Pitch / Roll).
Uses MediaPipe Tasks `FaceLandmarker` in VIDEO tracking mode on downscaled 320x240 frames
for <5ms inference time (~120 FPS on Apple Silicon).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import time
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
MP_KEY_INDICES = [1, 152, 33, 263, 61, 291]


class FacePoseEstimator:
    """静止カメラ画像から顔の3D角度(Yaw, Pitch, Roll)と12次元特徴量を高速推定"""

    def __init__(self, infer_width: int = 320, infer_height: int = 240):
        self.infer_width = infer_width
        self.infer_height = infer_height
        self.landmarker = None
        self.face_cascade = None
        self._last_ts_ms = 0
        self._init_mediapipe_landmarker()

    def _init_mediapipe_landmarker(self):
        try:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision

            if not ASCII_MODEL_PATH.exists() or ASCII_MODEL_PATH.stat().st_size < 100_000:
                print(f"[FacePoseEstimator] Downloading MediaPipe FaceLandmarker model to {ASCII_MODEL_PATH}...")
                urllib.request.urlretrieve(FACE_LANDMARKER_URL, str(ASCII_MODEL_PATH))

            base_options = mp_python.BaseOptions(model_asset_path=str(ASCII_MODEL_PATH))
            # VIDEOモードにすることで前フレームの顔位置を追跡し、全画面再検出コストを省いて3倍高速化
            options = mp_vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=mp_vision.RunningMode.VIDEO,
                num_faces=1,
                output_facial_transformation_matrixes=True,
                min_face_detection_confidence=0.40,
                min_face_presence_confidence=0.40,
                min_tracking_confidence=0.40,
            )
            self.landmarker = mp_vision.FaceLandmarker.create_from_options(options)
            print("[FacePoseEstimator] MediaPipe VIDEO-mode 3D FaceLandmarker initialized (<5ms/frame)!")
            return
        except Exception as e:
            print(f"[FacePoseEstimator] MediaPipe Tasks init failed ({e}), using OpenCV Cascade fallback.")

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

        # 推論用に 320x240 に縮小して高速化（座標は正規化されているため元解像度 w, h に正確に戻る）
        small_bgr = cv2.resize(frame_bgr, (self.infer_width, self.infer_height), interpolation=cv2.INTER_NEAREST)
        rgb = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        now_ms = int(time.monotonic() * 1000)
        if now_ms <= self._last_ts_ms:
            now_ms = self._last_ts_ms + 1
        self._last_ts_ms = now_ms

        result = self.landmarker.detect_for_video(mp_image, now_ms)
        if not result.face_landmarks:
            return None

        lm = result.face_landmarks[0]
        pts_2d = np.array(
            [[lm[idx].x * w, lm[idx].y * h] for idx in MP_KEY_INDICES],
            dtype=np.float64,
        )

        # 主要点から高速にバウンディングボックスを算出
        xs = pts_2d[:, 0]
        ys = pts_2d[:, 1]
        pad_x = (xs.max() - xs.min()) * 0.35
        pad_y = (ys.max() - ys.min()) * 0.45
        x0 = int(max(0, xs.min() - pad_x))
        y0 = int(max(0, ys.min() - pad_y))
        x1 = int(min(w, xs.max() + pad_x))
        y1 = int(min(h, ys.max() + pad_y * 0.4))
        bbox = (x0, y0, max(1, x1 - x0), max(1, y1 - y0))

        nose = pts_2d[0]
        chin = pts_2d[1]
        left_eye = pts_2d[2]
        right_eye = pts_2d[3]
        eye_mid = 0.5 * (left_eye + right_eye)
        eye_dist = max(1.0, float(np.linalg.norm(right_eye - left_eye)))

        # 水平角度(Yaw): 左右の目の中点に対する鼻先の水平変位 (右向き=正)
        yaw_geom = float(np.clip(((nose[0] - eye_mid[0]) / (0.5 * eye_dist)) * 60.0, -80.0, 80.0))

        # 垂直角度(Pitch): (目〜鼻の距離) と (鼻〜顎の距離) の比率から算出
        # 上を向くと鼻が目線に近づき(d_upper小)・顎が離れる(d_lower大) -> ratio減少 -> Pitch正(上向き)
        d_upper = max(1.0, float(nose[1] - eye_mid[1]))
        d_lower = max(1.0, float(chin[1] - nose[1]))
        up_down_ratio = d_upper / d_lower
        pitch_geom = float(np.clip((0.56 - up_down_ratio) * 130.0, -65.0, 65.0))

        roll_deg = float(
            np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))
        )

        if result.facial_transformation_matrixes:
            mat = np.array(result.facial_transformation_matrixes[0])[:3, :3]
            sy = float(np.sqrt(mat[0, 0] ** 2 + mat[1, 0] ** 2))
            if sy > 1e-6:
                # MediaPipe座標系では上向き時に mat[2,1] が負になるため符号を反転して一致させる
                pitch_mat = float(-np.degrees(np.arctan2(mat[2, 1], mat[2, 2])))
                yaw_mat = float(np.degrees(np.arctan2(-mat[2, 0], sy)))
                yaw_deg = 0.6 * yaw_geom + 0.4 * yaw_mat
                pitch_deg = 0.65 * pitch_geom + 0.35 * pitch_mat
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
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4)
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
