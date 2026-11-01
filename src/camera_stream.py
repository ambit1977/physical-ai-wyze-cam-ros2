"""
Ultra-low-latency Camera Stream helper supporting ATOM Cam 2 (RTSP), Mac Webcam, and Wyze Cam RTSP.

高速化・低遅延対策:
  1. RTSP時は `grab()` のみを高速ループで回して古いパケットを即座に破棄し、
     `read()` が呼ばれた瞬間だけデコード (`retrieve()`) することでデコード渋滞をゼロにする
  2. FFmpegの `probesize=32`, `analyzeduration=0`, `fflags=nobuffer`, `flags=low_delay` を適用
  3. フレームID (`frame_id`) を管理し、同一フレームに対する重複AI推論をスキップ可能にする
"""

import os
import threading
import time
from typing import Optional, Tuple, Union

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;udp|fflags;nobuffer|flags;low_delay|probesize;32|analyzeduration;0|max_delay;0"
)

import cv2
import numpy as np


class CameraStream:
    """RTSPストリームまたはUSB/内蔵カメラから最小遅延で最新フレームを取得するクラス"""

    def __init__(self, source: Union[str, int] = "0", width: int = 640, height: int = 480):
        if isinstance(source, str) and source.isdigit():
            self.source: Union[str, int] = int(source)
            self.is_rtsp = False
        else:
            self.source = source
            self.is_rtsp = isinstance(source, str) and source.lower().startswith("rtsp://")

        self.width = width
        self.height = height
        self.cap: Optional[cv2.VideoCapture] = None
        self.latest_frame: Optional[np.ndarray] = None
        self.frame_id: int = 0
        self.last_read_id: int = -1
        self.lock = threading.Lock()
        self.running = False
        self.thread: Optional[threading.Thread] = None

        self._open()

    def _open(self):
        if self.is_rtsp:
            self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
            if not self.cap.isOpened():
                # UDPで繋がらない場合はTCPにフォールバック
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                    "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|probesize;32|analyzeduration;0|max_delay;0"
                )
                self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        else:
            self.cap = cv2.VideoCapture(self.source)

        if self.cap is not None and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not self.is_rtsp:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self.cap.set(cv2.CAP_PROP_FPS, 30)

            ret, frame = self.cap.read()
            if ret and frame is not None:
                if frame.shape[1] != self.width or frame.shape[0] != self.height:
                    frame = cv2.resize(frame, (self.width, self.height))
                self.latest_frame = frame
                self.frame_id = 1

            self.running = True
            self.thread = threading.Thread(target=self._reader_loop, daemon=True)
            self.thread.start()

    def _reader_loop(self):
        while self.running and self.cap is not None:
            if not self.cap.isOpened():
                time.sleep(0.05)
                continue

            ret, frame = self.cap.read()
            if not ret or frame is None:
                time.sleep(0.005)
                continue

            if frame.shape[1] != self.width or frame.shape[0] != self.height:
                frame = cv2.resize(frame, (self.width, self.height))

            with self.lock:
                self.latest_frame = frame
                self.frame_id += 1

    def read(self) -> Optional[np.ndarray]:
        with self.lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame

    def read_new(self) -> Tuple[Optional[np.ndarray], bool]:
        """(最新フレーム, 前回呼び出し以降に更新されたか) を返す"""
        with self.lock:
            if self.latest_frame is None:
                return None, False
            is_new = self.frame_id != self.last_read_id
            self.last_read_id = self.frame_id
            return self.latest_frame, is_new

    def release(self):
        self.running = False
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=0.5)
        if self.cap is not None:
            self.cap.release()
