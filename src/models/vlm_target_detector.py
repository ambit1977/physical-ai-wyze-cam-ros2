"""
VLM & Open-Vocabulary Target Detector for directing Wyze Cam Pan v3 toward a specified object.

Supports 3 backends selectable via `--backend`:
  1. `owlv2`     : Hugging Face `google/owlv2-base-patch16-ensemble` (Open-Vocabulary Zero-Shot Detection)
                   任意の自然言語（例: "red coffee mug", "remote control", "stuffed toy"）で検出可能！
  2. `moondream` : Hugging Face `vikhyatk/moondream2` (Lightweight 2B VLM with Pointing/Grounding)
                   自然言語指示（例: "the person's face", "black bottle on the left"）で座標を指差し！
  3. `yolo`      : Ultralytics YOLOv8 (`yolov8n.pt`) 高速リアルタイム検出 (COCO 80クラス)
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import cv2
import numpy as np


@dataclass
class TargetDetection:
    label: str
    confidence: float
    center_norm: Tuple[float, float]  # (cx_norm, cy_norm) in [-0.5, +0.5] (画像中心が 0, 0)
    bbox: Tuple[int, int, int, int]   # (x, y, w, h) in pixels


class VLMTargetDetector:
    """
    指定されたテキスト指示（物体名または自然言語プロンプト）から、
    画像内のターゲット位置を推論し、カメラ中心からの角度偏差を出すクラス。
    """

    def __init__(self, backend: str = "owlv2", device: Optional[str] = None):
        self.backend = backend.lower()
        self.device = device or self._auto_device()
        self.model = None
        self.processor = None
        self._init_backend()

    @staticmethod
    def _auto_device() -> str:
        try:
            import torch

            if torch.backends.mps.is_available():
                return "mps"
            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"

    def _init_backend(self):
        if self.backend == "owlv2":
            try:
                from transformers import pipeline

                model_id = "google/owlv2-base-patch16-ensemble"
                print(f"[VLMTargetDetector] Loading Hugging Face Open-Vocabulary detector: {model_id} on {self.device}...")
                self.model = pipeline(
                    task="zero-shot-object-detection",
                    model=model_id,
                    device=self.device,
                )
                return
            except Exception as e:
                print(f"[VLMTargetDetector] OWLv2 load failed ({e}), falling back to YOLOv8...")
                self.backend = "yolo"

        if self.backend == "moondream":
            try:
                import torch
                from transformers import AutoModelForCausalLM

                model_id = "vikhyatk/moondream2"
                print(f"[VLMTargetDetector] Loading Hugging Face VLM: {model_id} on {self.device}...")
                dtype = torch.float16 if self.device in ("mps", "cuda") else torch.float32
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    trust_remote_code=True,
                    torch_dtype=dtype,
                ).to(self.device)
                self.model.eval()
                return
            except Exception as e:
                print(f"[VLMTargetDetector] Moondream2 load failed ({e}), falling back to YOLOv8...")
                self.backend = "yolo"

        if self.backend == "yolo":
            try:
                from ultralytics import YOLOWorld

                print("[VLMTargetDetector] Loading YOLO-World (CLIP Open-Vocabulary) detector...")
                self.model = YOLOWorld("yolov8s-world.pt")
                self._active_target_key = ""
                self._target_synonyms = {
                    "cup": ["cup", "coffee mug"],
                    "plastic bottle": ["plastic bottle", "water bottle", "bottle"],
                    "cell phone": ["cell phone", "smartphone"],
                    "milk carton": ["milk carton", "paper beverage carton"],
                }
                self._distractor_classes = [
                    "room wall",
                    "ceiling",
                    "floor",
                    "desk table",
                    "computer monitor",
                    "curtain",
                    "human face",
                ]
                self._update_yolo_world_classes("cup")
            except Exception as e:
                print(f"[VLMTargetDetector] YOLO-World fallback to YOLOv8n ({e})...")
                try:
                    from ultralytics import YOLO

                    self.model = YOLO("yolov8n.pt")
                    self._active_target_key = None
                except Exception as e2:
                    print(f"[VLMTargetDetector] YOLOv8 not available ({e2}), using color fallback.")
                    self.backend = "color"

    def _update_yolo_world_classes(self, target_lower: str):
        if getattr(self, "_active_target_key", None) == target_lower:
            return
        self._active_target_key = target_lower
        pos_classes = self._target_synonyms.get(target_lower, [target_lower])
        self._current_pos_set = set(pos_classes)
        all_classes = pos_classes + self._distractor_classes
        self.model.set_classes(all_classes)

    def detect(
        self, frame_bgr: np.ndarray, target_prompt: str, conf_threshold: float = 0.35
    ) -> Optional[TargetDetection]:
        h, w = frame_bgr.shape[:2]

        if self.backend == "owlv2" and self.model is not None:
            return self._detect_owlv2(frame_bgr, target_prompt, conf_threshold, w, h)
        elif self.backend == "moondream" and self.model is not None:
            return self._detect_moondream(frame_bgr, target_prompt, w, h)
        elif self.backend == "yolo" and self.model is not None:
            return self._detect_yolo(frame_bgr, target_prompt, conf_threshold, w, h)
        else:
            return self._detect_color_fallback(frame_bgr, target_prompt, w, h)

    def _detect_owlv2(
        self, frame_bgr: np.ndarray, target_prompt: str, conf_threshold: float, w: int, h: int
    ) -> Optional[TargetDetection]:
        from PIL import Image

        pil_img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        candidate_labels = [s.strip() for s in target_prompt.split(",") if s.strip()]
        preds = self.model(pil_img, candidate_labels=candidate_labels, threshold=conf_threshold)
        if not preds:
            return None

        best = max(preds, key=lambda p: p["score"])
        box = best["box"]
        x0, y0, x1, y1 = int(box["xmin"]), int(box["ymin"]), int(box["xmax"]), int(box["ymax"])
        bw, bh = max(1, x1 - x0), max(1, y1 - y0)
        cx_norm = (x0 + 0.5 * bw) / float(w) - 0.5
        cy_norm = (y0 + 0.5 * bh) / float(h) - 0.5
        return TargetDetection(
            label=best["label"],
            confidence=float(best["score"]),
            center_norm=(cx_norm, cy_norm),
            bbox=(x0, y0, bw, bh),
        )

    def _detect_moondream(
        self, frame_bgr: np.ndarray, target_prompt: str, w: int, h: int
    ) -> Optional[TargetDetection]:
        from PIL import Image

        pil_img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        # Moondream2 の point() または detect() メソッドを使用
        if hasattr(self.model, "detect"):
            out = self.model.detect(pil_img, target_prompt)
            objs = out.get("objects", [])
            if objs:
                b = objs[0]
                x0 = int(b["x_min"] * w)
                y0 = int(b["y_min"] * h)
                x1 = int(b["x_max"] * w)
                y1 = int(b["y_max"] * h)
                bw, bh = max(1, x1 - x0), max(1, y1 - y0)
                cx_norm = (x0 + 0.5 * bw) / float(w) - 0.5
                cy_norm = (y0 + 0.5 * bh) / float(h) - 0.5
                return TargetDetection(
                    label=target_prompt,
                    confidence=0.90,
                    center_norm=(cx_norm, cy_norm),
                    bbox=(x0, y0, bw, bh),
                )
        if hasattr(self.model, "point"):
            out = self.model.point(pil_img, target_prompt)
            pts = out.get("points", [])
            if pts:
                px = float(pts[0]["x"])
                py = float(pts[0]["y"])
                cx, cy = int(px * w), int(py * h)
                bw, bh = int(0.15 * w), int(0.15 * h)
                return TargetDetection(
                    label=target_prompt,
                    confidence=0.90,
                    center_norm=(px - 0.5, py - 0.5),
                    bbox=(max(0, cx - bw // 2), max(0, cy - bh // 2), bw, bh),
                )
        return None

    def _detect_yolo(
        self, frame_bgr: np.ndarray, target_prompt: str, conf_threshold: float, w: int, h: int
    ) -> Optional[TargetDetection]:
        target_lower = target_prompt.lower().strip()
        if getattr(self, "_active_target_key", None) is not None:
            self._update_yolo_world_classes(target_lower)

        results = self.model.predict(frame_bgr, conf=conf_threshold, verbose=False)
        if not results or len(results[0].boxes) == 0:
            return None

        boxes = results[0].boxes
        names = results[0].names
        pos_set = getattr(self, "_current_pos_set", {target_lower})

        best_det: Optional[TargetDetection] = None
        for box in boxes:
            cls_id = int(box.cls[0].item())
            cls_name = str(names.get(cls_id, "")).lower()
            conf = float(box.conf[0].item())
            if target_lower not in ("any", "*"):
                if cls_name not in pos_set and target_lower not in cls_name and cls_name not in target_lower:
                    continue
            x0, y0, x1, y1 = [int(v) for v in box.xyxy[0].tolist()]
            bw, bh = max(1, x1 - x0), max(1, y1 - y0)
            # 画面の45%以上を占める巨大なバウンディングボックスは壁・机などの背景誤検出として除外
            if (bw * bh) > 0.45 * (w * h):
                continue
            cx_norm = (x0 + 0.5 * bw) / float(w) - 0.5
            cy_norm = (y0 + 0.5 * bh) / float(h) - 0.5
            det = TargetDetection(
                label=target_lower,
                confidence=conf,
                center_norm=(cx_norm, cy_norm),
                bbox=(x0, y0, bw, bh),
            )
            if best_det is None or det.confidence > best_det.confidence:
                best_det = det
        return best_det

    def _detect_color_fallback(
        self, frame_bgr: np.ndarray, target_prompt: str, w: int, h: int
    ) -> Optional[TargetDetection]:
        """モデル未ロード環境用の高彩度物体フォールバック検出"""
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (0, 120, 100), (180, 255, 255))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < 400:
            return None
        x, y, bw, bh = cv2.boundingRect(largest)
        cx_norm = (x + 0.5 * bw) / float(w) - 0.5
        cy_norm = (y + 0.5 * bh) / float(h) - 0.5
        return TargetDetection(
            label=target_prompt,
            confidence=0.5,
            center_norm=(cx_norm, cy_norm),
            bbox=(x, y, bw, bh),
        )
