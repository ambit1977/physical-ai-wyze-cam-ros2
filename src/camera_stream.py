"""
Ultra-low-latency Camera Stream helper supporting ATOM Cam 2 (RTSP), Mac Webcam, and Wyze Cam RTSP.

RTSP遅延対策:
  1. FFmpeg Capture Options (`fflags=nobuffer`, `flags=low_delay`, `max_delay=0`) を自動適用
  2. バックグラウンドスレッドで常時フレームを破棄(grab)し続け、AI推論側が `read()` を呼んだ瞬間の
     「最新フレーム1枚だけ」を返すことで、数秒のバッファ滞留ラグを解消する。
"""

import os
import threading
import time
from typing import Optional, Union

# OpenCV の VideoCapture インポート前に FFmpeg の低遅延オプションを設定
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|framedrop;1|max_delay;0",
)

import cv2
import numpy as np


class CameraStream:
    """RTSPストリームまたはUSB/内蔵カメラから常に最新フレームのみを低遅延で取得するラッパー"""

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
        self.lock = threading.Lock()
        self.running = False
        self.thread: Optional[threading.Thread] = None

        self._open()

    def _open(self):
        if self.is_rtsp:
            self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        else:
            self.cap = cv2.VideoCapture(self.source)

        if self.cap is not None and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not self.is_rtsp:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

            # 初回フレーム取得
            ret, frame = self.cap.read()
            if ret and frame is not None:
                self.latest_frame = frame

            # RTSPのフレーム滞留による遅延（2〜5秒）を防ぐため、バックグラウンドで常時最新フレームに更新
            self.running = True
            self.thread = threading.Thread(target=self._reader_loop, daemon=True)
            self.thread.start()

    def _reader_loop(self):
        """常にストリームからフレームを引き抜き、古いバッファを捨てて最新フレームだけ保持する"""
        while self.running and self.cap is not None:
            if not self.cap.isOpened():
                time.sleep(0.1)
                continue
            # grab() で高速にバッファを消化し、最新のみ retrieve()
            grabbed = self.cap.grab()
            if not grabbed:
                time.sleep(0.01)
                continue
            ret, frame = self.cap.retrieve()
            if ret and frame is not None:
                if self.is_rtsp and (frame.shape[1] != self.width or frame.shape[0] != self.height):
                    frame = cv2.resize(frame, (self.width, self.height))
                with self.lock:
                    self.latest_frame = frame

    def read(self) -> Optional[np.ndarray]:
        with self.lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

    def release(self):
        self.running = False
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()
