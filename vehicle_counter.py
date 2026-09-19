"""
vehicle_counter.py
------------------
Contador de veículos baseado em detecção YOLO sem ByteTrack (fallback).
Usa o VehicleTracker interno baseado em distância euclidiana.
Mantido para compatibilidade; o pipeline principal usa video_processor.py.
"""

from __future__ import annotations

import queue
import threading
import time
from time import sleep
from typing import List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from vehicle_tracker import VehicleTracker, Detection

# ---------------------------------------------------------------------------
# Estado global
# ---------------------------------------------------------------------------
_count: int = 0
_count_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Utilitários
# ---------------------------------------------------------------------------

def _center(x1: int, y1: int, x2: int, y2: int) -> Tuple[int, int]:
    return (x1 + x2) // 2, (y1 + y2) // 2


def _detect_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            mem = torch.cuda.get_device_properties(0).total_memory / 1024 ** 2
            return "0" if mem >= 2000 else "cpu"
    except Exception:
        pass
    return "cpu"


# ---------------------------------------------------------------------------
# Processador manual (sem ByteTrack)
# ---------------------------------------------------------------------------

class ManualVideoProcessor:
    """
    Pipeline simples com YOLO predict + VehicleTracker euclidiano.
    Ideal para ambientes sem suporte ao ByteTrack.
    """

    _VEHICLE_CLASSES = {2, 5, 7}
    _INFER_WIDTH = 640
    _QUEUE_SIZE = 2

    def __init__(self, video_source: str) -> None:
        global _count
        _count = 0

        self.video_source = video_source
        self.running = True
        self.device = _detect_device()

        self._capture_q: queue.Queue = queue.Queue(maxsize=self._QUEUE_SIZE)
        self._display_q: queue.Queue = queue.Queue(maxsize=self._QUEUE_SIZE)

        self._tracker = VehicleTracker()
        self._setup_capture()
        self._setup_model()

        self._line_y = int(self._roi_h * 0.75)

        self._infer_interval = 1 / 15 if self.device != "cpu" else 1 / 10
        self._last_infer = 0.0

        self._fps = 0.0
        self._frame_count = 0
        self._fps_start = time.time()

    # ------------------------------------------------------------------

    def _open_cap(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self.video_source, cv2.CAP_FFMPEG)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        return cap

    def _setup_capture(self) -> None:
        self.cap = self._open_cap()
        if not self.cap.isOpened():
            raise RuntimeError(f"Não foi possível abrir: {self.video_source}")
        ret, frame = self.cap.read()
        if not ret or frame is None:
            raise RuntimeError("Não foi possível ler o primeiro frame.")
        h, w = frame.shape[:2]
        self._roi_w = int(w * 0.60)
        self._roi_h = h
        self._roi_x = (w - self._roi_w) // 2
        self._roi_y = 0
        print(f"[Manual] ROI: {self._roi_w}x{self._roi_h}")

    def _setup_model(self) -> None:
        print("[Manual] Carregando YOLO…")
        self.model = YOLO("yolov8n.pt")
        self.model.overrides.update({"verbose": False, "device": self.device,
                                     "half": self.device != "cpu"})
        print(f"[Manual] Dispositivo: {self.device}")

    # ------------------------------------------------------------------
    # Threads
    # ------------------------------------------------------------------

    def _capture_loop(self) -> None:
        fails = 0
        while self.running:
            ret, full = self.cap.read()
            if not ret or full is None:
                fails += 1
                sleep(min(fails * 0.5, 5.0))
                self.cap.release()
                self.cap = self._open_cap()
                continue
            fails = 0
            roi = full[self._roi_y:self._roi_y + self._roi_h,
                       self._roi_x:self._roi_x + self._roi_w]
            self._drop_put(self._capture_q, roi.copy())
            sleep(1 / 25)

    def _process_loop(self) -> None:
        while self.running:
            now = time.time()
            if now - self._last_infer < self._infer_interval:
                sleep(0.005)
                continue
            try:
                frame = self._capture_q.get(timeout=0.1)
            except queue.Empty:
                continue
            detections = self._detect(frame)
            rendered = self._render(frame, detections)
            self._drop_put(self._display_q, rendered)
            self._last_infer = time.time()

    # ------------------------------------------------------------------

    def _detect(self, frame: np.ndarray) -> List[Detection]:
        try:
            h, w = frame.shape[:2]
            scale = self._INFER_WIDTH / w if w > self._INFER_WIDTH else 1.0
            small = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale < 1 else frame

            results = self.model.predict(
                small, verbose=False, conf=0.15, iou=0.45,
                device=self.device, classes=list(self._VEHICLE_CLASSES),
            )

            detections: List[Detection] = []
            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    x1, y1, x2, y2 = (int(v / scale) for v in map(float, box.xyxy[0]))
                    if (x2 - x1) < 18 or (y2 - y1) < 18:
                        continue
                    detections.append((_center(x1, y1, x2, y2), (x1, y1, x2, y2)))
            return detections
        except Exception as exc:
            print(f"[Detecção] Erro: {exc}")
            return []

    def _render(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        global _count
        out = frame.copy()
        crossed = self._tracker.count_crossings(detections, self._line_y)

        if crossed:
            with _count_lock:
                _count += crossed
            print(f"[Contador] +{crossed} | Total: {_count}")

        # Linha
        color = (0, 255, 0) if crossed else (0, 220, 220)
        cv2.line(out, (0, self._line_y), (out.shape[1], self._line_y), color, 3)

        # Detecções
        for (cx, cy), (x1, y1, x2, y2) in detections:
            cv2.rectangle(out, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.circle(out, (cx, cy), 4, (0, 0, 255), -1)

        # HUD
        with _count_lock:
            cv2.putText(out, f"TOTAL: {_count}", (20, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.putText(out, f"FPS: {self._fps:.1f}", (20, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(out, f"Tracks: {len(self._tracker.get_tracked_vehicles())}", (20, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        if crossed:
            cv2.putText(out, f"+{crossed} PASSOU", (20, 260),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3)
        return out

    # ------------------------------------------------------------------

    @staticmethod
    def _drop_put(q: queue.Queue, item) -> None:
        while not q.empty():
            try:
                q.get_nowait()
            except queue.Empty:
                break
        try:
            q.put_nowait(item)
        except queue.Full:
            pass

    # ------------------------------------------------------------------

    def run(self) -> None:
        cap_t = threading.Thread(target=self._capture_loop, daemon=True)
        proc_t = threading.Thread(target=self._process_loop, daemon=True)
        cap_t.start()
        proc_t.start()
        print("[Manual] Sistema iniciado. Pressione ESC ou Q para sair.")

        try:
            while self.running:
                try:
                    frame = self._display_q.get(timeout=0.05)
                except queue.Empty:
                    frame = None

                if frame is not None:
                    self._frame_count += 1
                    elapsed = time.time() - self._fps_start
                    if elapsed >= 2.0:
                        self._fps = self._frame_count / elapsed
                        self._frame_count = 0
                        self._fps_start = time.time()
                    cv2.imshow("Fluxo Veicular (Manual)", frame)

                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
                sleep(0.001)

        except KeyboardInterrupt:
            pass
        finally:
            self.running = False
            self.cap.release()
            cv2.destroyAllWindows()
            print(f"Encerrado. Total: {_count} veículos.")
