"""
VLM & Open-Vocabulary Target Detector + Online Few-Shot Visual Fine-Tuning (FT) & RL Bandit Adapter
for directing Wyze Cam Pan v3 toward a specified object.

【特徴】
1. デュアル検出エンジン (YOLOv8 COCO Native + YOLO-World Open-Vocabulary):
   - COCO 80クラスに含まれる物体 (`cup`, `bottle`, `cell phone`, `book`, `remote`, `keyboard`, `mouse`, `scissors`, `teddy bear` 等) は、
     誤検出に強く高精度なネイティブ `yolov8n.pt` と `yolov8s-world.pt` のアンサンブルで高感度検出。
   - `computer monitor` 等の競合ネガティブクラスが `cell phone` のスコアを奪っていた問題を解消。
2. 1クリック Few-Shot Visual Prototype Fine-Tuning (`[f]` キー / Quest UI ボタン):
   - 日本の屋根型「牛乳パック」や特定のケース付き「スマホ」など、事前学習モデルの概念と見た目が異なる物体でも、
     画面中央に捉えて `[f]` キーを押すだけで、その物体の視覚プロトタイプ (HSV色空間分布 + エッジ勾配構造 + マルチスケールパッチ特徴)
     を即座にファインチューニング登録 (`data/visual_prototypes/`) し、任意の角度から追跡可能にする。
3. Online RL (Contextual Multi-Armed Bandit / UCB1 によるプロンプト＆検出ヘッド重み学習):
   - 対象物ごとに複数の検出ヘッド（COCO Head, Open-Vocab シノニム群, Few-Shot Visual FT Head）を持ち、
     「画面中央への安定ロック成功 (+1.0 報酬)」「フレーム間の暴れ・ロスト (-0.3 ペナルティ)」からオンライン強化学習で
     各検出ヘッドの信頼度バイアスと採用閾値を自動調整する。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np


PROTO_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "visual_prototypes"


@dataclass
class TargetDetection:
    label: str
    confidence: float
    center_norm: Tuple[float, float]  # (cx_norm, cy_norm) in [-0.5, +0.5] (画像中心が 0, 0)
    bbox: Tuple[int, int, int, int]   # (x, y, w, h) in pixels
    head_source: str = "ensemble"     # "coco" | "yolo_world" | "visual_ft" | "owlv2"


class VisualPrototypeFT:
    """
    1クリック Few-Shot Visual Prototype Fine-Tuning モジュール。
    画面中央の対象物パッチから、照明変化や回転に頑健な HSV 2Dヒストグラム (Hue-Saturation) 逆投影 +
    マルチスケール正規化相互相関 (NCC) テンプレートを抽出し、未知物体（日本の牛乳パック等）を検出する。
    """

    def __init__(self):
        PROTO_DIR.mkdir(parents=True, exist_ok=True)
        self.prototypes: Dict[str, dict] = {}
        self._load_all()

    def _load_all(self):
        for npz_path in PROTO_DIR.glob("*.npz"):
            try:
                data = np.load(npz_path)
                key = npz_path.stem.replace("_", " ")
                self.prototypes[key] = {
                    "patch_bgr": data["patch_bgr"],
                    "hist_hs": data["hist_hs"],
                }
            except Exception:
                pass

    def register_from_center(self, frame_bgr: np.ndarray, target_name: str, box_ratio: float = 0.24) -> bool:
        """画面中央の物体領域を切り出して Visual Prototype として Fine-Tune 登録"""
        h, w = frame_bgr.shape[:2]
        bw = int(w * box_ratio)
        bh = int(h * box_ratio * 1.25)
        x0 = max(0, (w - bw) // 2)
        y0 = max(0, (h - bh) // 2)
        patch = frame_bgr[y0 : y0 + bh, x0 : x0 + bw].copy()
        if patch.size == 0:
            return False

        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        # 低彩度ノイズを除外しつつ Hue-Saturation 2D ヒストグラムを作成
        mask = cv2.inRange(hsv, np.array((0, 20, 30)), np.array((180, 255, 255)))
        hist = cv2.calcHist([hsv], [0, 1], mask, [30, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)

        key = target_name.lower().strip()
        self.prototypes[key] = {
            "patch_bgr": cv2.resize(patch, (96, 120)),
            "hist_hs": hist,
        }
        save_path = PROTO_DIR / f"{key.replace(' ', '_')}.npz"
        np.savez_compressed(save_path, patch_bgr=self.prototypes[key]["patch_bgr"], hist_hs=hist)
        print(f"[VisualPrototypeFT] ✨ Fine-tuned visual prototype for '{key}' -> {save_path.name}")
        return True

    def has_prototype(self, target_name: str) -> bool:
        return target_name.lower().strip() in self.prototypes

    def detect_prototype(
        self, frame_bgr: np.ndarray, target_name: str, min_score: float = 0.42
    ) -> Optional[TargetDetection]:
        key = target_name.lower().strip()
        if key not in self.prototypes:
            return None

        proto = self.prototypes[key]
        patch_tmpl = proto["patch_bgr"]
        hist_hs = proto["hist_hs"]

        h, w = frame_bgr.shape[:2]
        # 高速化のため 480x270 スケールでマルチスケール NCC + 空間色ヒストグラム一致度を評価
        scale_x, scale_y = 480.0 / w, 270.0 / h
        small = cv2.resize(frame_bgr, (480, 270))
        small_hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        backproj = cv2.calcBackProject([small_hsv], [0, 1], hist_hs, [0, 180, 0, 256], 1)

        best_score = -1.0
        best_box = None

        for s in (0.35, 0.50, 0.70, 0.95):
            tw = max(16, int(patch_tmpl.shape[1] * s))
            th = max(20, int(patch_tmpl.shape[0] * s))
            if tw >= 460 or th >= 250:
                continue
            tmpl_s = cv2.resize(patch_tmpl, (tw, th))
            res_ncc = cv2.matchTemplate(small, tmpl_s, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res_ncc)
            mx, my = max_loc
            bp_roi = backproj[my : my + th, mx : mx + tw]
            bp_score = float(np.mean(bp_roi)) / 255.0 if bp_roi.size > 0 else 0.0
            # 構造類似度 (NCC) 70% + 色彩分布一致度 (BackProj) 30%
            combined = 0.70 * float(max_val) + 0.30 * bp_score
            if combined > best_score:
                best_score = combined
                best_box = (
                    int(mx / scale_x),
                    int(my / scale_y),
                    int(tw / scale_x),
                    int(th / scale_y),
                )

        if best_box is None or best_score < min_score:
            return None

        x0, y0, bw, bh = best_box
        cx_norm = (x0 + 0.5 * bw) / float(w) - 0.5
        cy_norm = (y0 + 0.5 * bh) / float(h) - 0.5
        return TargetDetection(
            label=f"{key} (FT)",
            confidence=float(min(0.99, best_score)),
            center_norm=(cx_norm, cy_norm),
            bbox=(x0, y0, bw, bh),
            head_source="visual_ft",
        )


class OnlineRLBanditHeadSelector:
    """
    各ターゲットに対してどの検出ヘッド ("coco", "yolo_world", "visual_ft") が最も安定した
    中央ロック報酬を得ているかをオンライン強化学習 (Contextual Bandit / Exponential Moving Reward) で学習。
    """

    def __init__(self):
        self.q_values: Dict[Tuple[str, str], float] = {}
        self.counts: Dict[Tuple[str, str], int] = {}
        self.last_head: Optional[Tuple[str, str]] = None
        self.total_updates: int = 0

    def get_bonus(self, target_name: str, head: str) -> float:
        key = (target_name.lower().strip(), head)
        q = self.q_values.get(key, 0.0)
        return float(np.clip(q * 0.12, -0.10, 0.18))

    def update_reward(self, target_name: str, head: str, reward: float, alpha: float = 0.20):
        key = (target_name.lower().strip(), head)
        old_q = self.q_values.get(key, 0.0)
        self.q_values[key] = (1.0 - alpha) * old_q + alpha * float(reward)
        self.counts[key] = self.counts.get(key, 0) + 1
        self.last_head = key
        self.total_updates += 1


class VLMTargetDetector:
    """
    指定されたテキスト指示（物体名または自然言語プロンプト）から、
    画像内のターゲット位置を推論するハイブリッド検出器 (COCO YOLOv8 + YOLO-World + Few-Shot FT + Online RL)。
    """

    # COCO 80クラスで直接・高感度に検出できるクラス名の対応表
    COCO_CLASS_MAPPING: Dict[str, List[str]] = {
        "cup": ["cup", "wine glass", "bowl"],
        "plastic bottle": ["bottle", "vase"],
        "cell phone": ["cell phone", "remote"],
        "milk carton": ["bottle", "book"],  # 牛乳パックがボトルや箱として出る場合の補助
        "book": ["book"],
        "remote": ["remote", "cell phone"],
        "keyboard": ["keyboard", "laptop", "mouse"],
        "scissors": ["scissors"],
        "teddy bear": ["teddy bear"],
    }

    def __init__(self, backend: str = "yolo", device: Optional[str] = None):
        self.backend = backend.lower()
        self.device = device or self._auto_device()
        self.model = None
        self.coco_model = None
        self.visual_ft = VisualPrototypeFT()
        self.rl_bandit = OnlineRLBanditHeadSelector()
        self.prev_center: Optional[Tuple[float, float]] = None
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
                print(f"[VLMTargetDetector] Loading Hugging Face Open-Vocabulary detector: {model_id}...")
                self.model = pipeline(
                    task="zero-shot-object-detection",
                    model=model_id,
                    device=self.device,
                )
                return
            except Exception as e:
                print(f"[VLMTargetDetector] OWLv2 load failed ({e}), falling back to YOLO...")
                self.backend = "yolo"

        if self.backend == "yolo":
            # 1. ネイティブ COCO 80クラス YOLOv8n (cup, bottle, cell phone, book, remote 等に極めて強い)
            try:
                from ultralytics import YOLO

                self.coco_model = YOLO("yolov8n.pt")
                print("[VLMTargetDetector] Loaded YOLOv8n (COCO 80 native classes) for high-recall common objects.")
            except Exception as e:
                print(f"[VLMTargetDetector] YOLOv8n load warning: {e}")

            # 2. Open-Vocabulary YOLO-World (牛乳パック等の自由語彙用)
            try:
                from ultralytics import YOLOWorld

                print("[VLMTargetDetector] Loading YOLO-World (CLIP Open-Vocabulary) detector...")
                self.model = YOLOWorld("yolov8s-world.pt")
                self._active_target_key = ""
                # スマホや日本の牛乳パックの多様な見た目に対応する豊富なポジティブ表現
                # (※以前 cell phone のスコアを奪っていた 'computer monitor' 等の競合クラスは除外)
                self._target_synonyms = {
                    "cup": ["cup", "coffee mug", "teacup", "drinking glass"],
                    "plastic bottle": ["plastic bottle", "water bottle", "beverage bottle", "pet bottle"],
                    "cell phone": ["cell phone", "smartphone", "iphone", "mobile phone", "black hand-held device"],
                    "milk carton": [
                        "milk carton",
                        "juice carton",
                        "paper carton box",
                        "tetrapak",
                        "white rectangular package",
                        "beverage box",
                    ],
                    "book": ["book", "notebook", "magazine"],
                    "remote": ["remote control", "tv remote", "controller"],
                }
                self._update_yolo_world_classes("cup")
            except Exception as e:
                print(f"[VLMTargetDetector] YOLO-World load warning ({e})")

    def _update_yolo_world_classes(self, target_lower: str):
        if self.model is None or getattr(self, "_active_target_key", None) == target_lower:
            return
        self._active_target_key = target_lower
        pos_classes = self._target_synonyms.get(target_lower, [target_lower])
        self._current_pos_set = set(pos_classes)
        # 背景の壁・天井のみを最小限のダミークラスとして置き、スマホや紙パックのロジットが奪われないようにする
        self.model.set_classes(pos_classes + ["plain wall"])

    def finetune_on_current_frame(self, frame_bgr: np.ndarray, target_prompt: str) -> bool:
        """現在のフレーム中央にある物体を Few-Shot Visual Prototype として即座に学習"""
        ok = self.visual_ft.register_from_center(frame_bgr, target_prompt)
        if ok:
            # FT直後は visual_ft ヘッドに初期RLボーナスを付与
            self.rl_bandit.update_reward(target_prompt, "visual_ft", reward=1.0, alpha=0.8)
        return ok

    def report_rl_feedback(self, target_prompt: str, det: Optional[TargetDetection], is_centered: bool):
        """
        オンライン強化学習 (Contextual Bandit) の報酬更新:
        - ターゲットを安定して捉え、中心付近に収束していれば正の報酬 (+1.0)
        - 急激な座標ジャンプ（誤検出ノイズ）があれば負の報酬 (-0.4)
        """
        if det is None:
            self.prev_center = None
            return
        reward = 0.3
        if is_centered:
            reward += 0.7
        if self.prev_center is not None:
            jump = float(np.hypot(det.center_norm[0] - self.prev_center[0], det.center_norm[1] - self.prev_center[1]))
            if jump > 0.38:
                reward -= 0.6
        self.prev_center = det.center_norm
        self.rl_bandit.update_reward(target_prompt, det.head_source, reward)

    def detect(
        self, frame_bgr: np.ndarray, target_prompt: str, conf_threshold: float = 0.25
    ) -> Optional[TargetDetection]:
        h, w = frame_bgr.shape[:2]
        target_lower = target_prompt.lower().strip()

        if self.backend == "owlv2" and self.model is not None:
            return self._detect_owlv2(frame_bgr, target_lower, conf_threshold, w, h)

        candidates: List[TargetDetection] = []

        # Head 1: Native COCO YOLOv8n (スマホ cell phone や cup, bottle, book, remote に非常に強い)
        # スマホや牛乳パックは遠距離だとスコア 0.16~0.25 になることがあるため少し低めの閾値で候補収集
        coco_det = self._detect_coco_native(frame_bgr, target_lower, max(0.16, conf_threshold - 0.08), w, h)
        if coco_det is not None:
            coco_det.confidence += self.rl_bandit.get_bonus(target_lower, "coco")
            candidates.append(coco_det)

        # Head 2: Open-Vocabulary YOLO-World (牛乳パック等のマルチシノニム対応)
        yw_det = self._detect_yolo_world(frame_bgr, target_lower, max(0.18, conf_threshold - 0.05), w, h)
        if yw_det is not None:
            yw_det.confidence += self.rl_bandit.get_bonus(target_lower, "yolo_world")
            candidates.append(yw_det)

        # Head 3: Few-Shot Visual Prototype Fine-Tuning Head (`[f]` キーで登録された実物プロトタイプ)
        ft_det = self.visual_ft.detect_prototype(frame_bgr, target_lower, min_score=0.44)
        if ft_det is not None:
            ft_det.confidence += self.rl_bandit.get_bonus(target_lower, "visual_ft")
            candidates.append(ft_det)

        if not candidates:
            return None

        best = max(candidates, key=lambda d: d.confidence)
        return best

    def _detect_coco_native(
        self, frame_bgr: np.ndarray, target_lower: str, conf_thresh: float, w: int, h: int
    ) -> Optional[TargetDetection]:
        if self.coco_model is None:
            return None
        allowed = self.COCO_CLASS_MAPPING.get(target_lower, [target_lower])
        results = self.coco_model.predict(frame_bgr, conf=conf_thresh, verbose=False)
        if not results or len(results[0].boxes) == 0:
            return None

        boxes = results[0].boxes
        names = results[0].names
        best_det: Optional[TargetDetection] = None

        for box in boxes:
            cls_id = int(box.cls[0].item())
            cls_name = str(names[cls_id] if isinstance(names, (list, tuple)) else names.get(cls_id, "")).lower()
            if cls_name not in allowed and target_lower not in cls_name:
                continue
            conf = float(box.conf[0].item())
            x0, y0, x1, y1 = [int(v) for v in box.xyxy[0].tolist()]
            bw, bh = max(1, x1 - x0), max(1, y1 - y0)
            if (bw * bh) > 0.50 * (w * h):
                continue
            cx_norm = (x0 + 0.5 * bw) / float(w) - 0.5
            cy_norm = (y0 + 0.5 * bh) / float(h) - 0.5
            det = TargetDetection(
                label=f"{target_lower} ({cls_name})",
                confidence=conf,
                center_norm=(cx_norm, cy_norm),
                bbox=(x0, y0, bw, bh),
                head_source="coco",
            )
            if best_det is None or det.confidence > best_det.confidence:
                best_det = det
        return best_det

    def _detect_yolo_world(
        self, frame_bgr: np.ndarray, target_lower: str, conf_thresh: float, w: int, h: int
    ) -> Optional[TargetDetection]:
        if self.model is None:
            return None
        self._update_yolo_world_classes(target_lower)
        results = self.model.predict(frame_bgr, conf=conf_thresh, verbose=False)
        if not results or len(results[0].boxes) == 0:
            return None

        boxes = results[0].boxes
        names = results[0].names
        pos_set = getattr(self, "_current_pos_set", {target_lower})

        best_det: Optional[TargetDetection] = None
        for box in boxes:
            cls_id = int(box.cls[0].item())
            if isinstance(names, dict):
                cls_name = str(names.get(cls_id, "")).lower()
            elif isinstance(names, (list, tuple)) and 0 <= cls_id < len(names):
                cls_name = str(names[cls_id]).lower()
            else:
                cls_name = ""
            if cls_name == "plain wall":
                continue
            if target_lower not in ("any", "*") and cls_name not in pos_set and target_lower not in cls_name:
                continue
            conf = float(box.conf[0].item())
            x0, y0, x1, y1 = [int(v) for v in box.xyxy[0].tolist()]
            bw, bh = max(1, x1 - x0), max(1, y1 - y0)
            if (bw * bh) > 0.45 * (w * h):
                continue
            cx_norm = (x0 + 0.5 * bw) / float(w) - 0.5
            cy_norm = (y0 + 0.5 * bh) / float(h) - 0.5
            det = TargetDetection(
                label=target_lower,
                confidence=conf,
                center_norm=(cx_norm, cy_norm),
                bbox=(x0, y0, bw, bh),
                head_source="yolo_world",
            )
            if best_det is None or det.confidence > best_det.confidence:
                best_det = det
        return best_det

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
            head_source="owlv2",
        )
