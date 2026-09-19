"""
video_processor.py
------------------
Processador de vídeo com YOLO + ByteTrack para contagem de fluxo veicular.
Arquitetura de 3 threads: captura → processamento → exibição.
"""

from __future__ import annotations

import queue
import threading
import time
from collections import defaultdict
from time import sleep
from typing import Dict, List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Tipos
# ---------------------------------------------------------------------------
BBox = Tuple[int, int, int, int]
Center = Tuple[int, int]
TrackedObject = Dict  # {track_id, bbox, center, class_id, conf}

# ---------------------------------------------------------------------------
# Estado global de contagem (protegido por lock)
# ---------------------------------------------------------------------------
_count_lock = threading.Lock()
_count_up: int = 0


def _reset_counts() -> None:
    global _count_up
    with _count_lock:
        _count_up = 0


# ---------------------------------------------------------------------------
# Utilitários
# ---------------------------------------------------------------------------

def _center_of_bbox(x1: int, y1: int, x2: int, y2: int) -> Center:
    return (x1 + x2) // 2, (y1 + y2) // 2


def _detect_device() -> str:
    """Retorna '0' (GPU) se disponível e com memória suficiente, senão 'cpu'."""
    try:
        import torch
        if not torch.cuda.is_available():
            print("CUDA não disponível – usando CPU.")
            return "cpu"
        name = torch.cuda.get_device_name(0)
        mem_mb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 2
        print(f"GPU: {name} ({mem_mb:.0f} MB)")
        return "0" if mem_mb >= 2000 else "cpu"
    except Exception as exc:
        print(f"Erro ao detectar GPU: {exc} – usando CPU.")
        return "cpu"


# ---------------------------------------------------------------------------
# FlowCounter
# ---------------------------------------------------------------------------

class FlowCounter:
    """
    Conta veículos que cruzam a linha virtual de baixo para cima (sentido
    prev_y > line_y  →  cy <= line_y).

    Atributos públicos
    ------------------
    line_y    : posição vertical da linha de contagem
    tolerance : espessura visual da zona da linha
    """

    def __init__(self, line_y: int, tolerance: int = 12) -> None:
        self.line_y = line_y
        self.tolerance = tolerance
        self._last_centers: Dict[int, Center] = {}
        self._counted_up: set = set()
        self._history: Dict[int, List[Center]] = defaultdict(list)
        self._idle_frames: Dict[int, int] = defaultdict(int)

    # ------------------------------------------------------------------

    def update(self, objects: List[TrackedObject]) -> int:
        """Processa objetos rastreados e retorna quantos cruzaram agora."""
        global _count_up

        new_crossings = 0

        for obj in objects:
            tid = obj["track_id"]
            cx, cy = obj["center"]

            # Histórico de trajetória (últimos 30 pontos)
            hist = self._history[tid]
            hist.append((cx, cy))
            if len(hist) > 30:
                hist.pop(0)

            prev = self._last_centers.get(tid)
            self._last_centers[tid] = (cx, cy)

            if prev is None:
                continue

            _, prev_y = prev
            if prev_y > self.line_y and cy <= self.line_y and tid not in self._counted_up:
                self._counted_up.add(tid)
                new_crossings += 1

        if new_crossings:
            with _count_lock:
                _count_up += new_crossings

        return new_crossings

    def get_history(self, track_id: int) -> List[Center]:
        return self._history.get(track_id, [])

    def cleanup_stale(self, active_ids: set, max_idle: int = 60) -> None:
        """Remove IDs inativos há mais de `max_idle` frames.

        Sem essa margem, qualquer oclusão de 1 frame apagava o histórico do
        ID e o `_counted_up` nunca era purgado, crescendo sem limite em
        sessões longas.
        """
        for tid in list(self._last_centers):
            if tid in active_ids:
                self._idle_frames[tid] = 0
                continue

            self._idle_frames[tid] += 1
            if self._idle_frames[tid] > max_idle:
                self._last_centers.pop(tid, None)
                self._history.pop(tid, None)
                self._counted_up.discard(tid)
                self._idle_frames.pop(tid, None)


# ---------------------------------------------------------------------------
# HUD
# ---------------------------------------------------------------------------

def _draw_hud(
    frame: np.ndarray,
    flow: FlowCounter,
    fps: float,
    num_tracks: int,
) -> None:
    """Desenha informações no frame (sem modificar estado)."""
    with _count_lock:
        total = _count_up

    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (300, 230), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

    items = [
        (f"TOTAL: {total}", (0, 255, 0), 1.2, 3),
        (f"TRACKS: {num_tracks}", (255, 255, 255), 0.8, 2),
        (f"FPS: {fps:.1f}", (255, 255, 255), 0.8, 2),
        (f"LINHA Y: {flow.line_y}", (200, 200, 200), 0.7, 1),
    ]
    y = 55
    for text, color, scale, thick in items:
        cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)
        y += int(scale * 45)


# ---------------------------------------------------------------------------
# SmoothVideoProcessor
# ---------------------------------------------------------------------------

class SmoothVideoProcessor:
    """
    Pipeline de 3 threads independentes:
      1. CaptureThread  – lê frames do stream e coloca na capture_queue
      2. ProcessThread  – executa YOLO + ByteTrack e coloca na display_queue
      3. Thread principal – exibe frames e trata entrada de teclado
    """

    # Configurações de fila
    _QUEUE_SIZE = 2
    # Classes YOLO: carro=2, ônibus=5, caminhão=7
    _VEHICLE_CLASSES = [2, 5, 7]
    # Largura máxima de inferência (equilibrio velocidade / precisão)
    _INFER_WIDTH = 960

    def __init__(self, video_source: str) -> None:
        _reset_counts()
        self.video_source = video_source
        self.running = True

        self._capture_q: queue.Queue = queue.Queue(maxsize=self._QUEUE_SIZE)
        self._display_q: queue.Queue = queue.Queue(maxsize=self._QUEUE_SIZE)

        self.device = _detect_device()
        self._setup_capture()
        self._setup_model()

        self.flow = FlowCounter(
            line_y=int(self._roi_h * 0.72),
            tolerance=12,
        )

        # Intervalo entre inferências
        self._infer_interval = 1 / 20 if self.device != "cpu" else 1 / 12
        self._last_infer: float = 0.0

        # FPS display
        self._frame_count = 0
        self._fps_start = time.time()
        self._fps = 0.0

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _open_capture(self, source: str) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        return cap

    def _setup_capture(self) -> None:
        self.cap = self._open_capture(self.video_source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Não foi possível abrir: {self.video_source}")

        ret, frame = self.cap.read()
        if not ret or frame is None:
            raise RuntimeError("Não foi possível ler o primeiro frame.")

        full_h, full_w = frame.shape[:2]
        print(f"Resolução original: {full_w}x{full_h}")

        # ROI: 60% central da largura, altura completa
        self._roi_w = int(full_w * 0.60)
        self._roi_h = full_h
        self._roi_x = (full_w - self._roi_w) // 2
        self._roi_y = 0
        print(f"ROI: {self._roi_w}x{self._roi_h} (x={self._roi_x})")

    def _setup_model(self) -> None:
        print("Carregando modelo YOLO…")
        self.model = YOLO("yolov8n.pt")
        self.model.overrides["verbose"] = False
        self.model.overrides["device"] = self.device
        self.model.overrides["half"] = self.device != "cpu"

        if self.device != "cpu":
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            self.model.track(dummy, persist=True, verbose=False, tracker="bytetrack.yaml")
            print("GPU aquecida.")

        print(f"Modelo pronto – dispositivo: {self.device}")

    # ------------------------------------------------------------------
    # Thread de captura
    # ------------------------------------------------------------------

    def _capture_loop(self) -> None:
        print("[Captura] Thread iniciada.")
        fails = 0
        sleep_s = 1 / 30 if self.device != "cpu" else 1 / 20

        while self.running:
            ret, full = self.cap.read()

            if not ret or full is None:
                fails += 1
                wait = min(fails * 0.5, 5.0)
                print(f"[Captura] Falha #{fails}, reconectando em {wait:.1f}s…")
                sleep(wait)
                self.cap.release()
                self.cap = self._open_capture(self.video_source)
                continue

            fails = 0
            roi = full[
                self._roi_y : self._roi_y + self._roi_h,
                self._roi_x : self._roi_x + self._roi_w,
            ]

            self._drop_and_put(self._capture_q, roi.copy())
            sleep(sleep_s)

    # ------------------------------------------------------------------
    # Thread de processamento
    # ------------------------------------------------------------------

    def _process_loop(self) -> None:
        print("[Processo] Thread iniciada.")

        while self.running:
            now = time.time()
            if now - self._last_infer < self._infer_interval:
                sleep(0.005)
                continue

            try:
                frame = self._capture_q.get(timeout=0.1)
            except queue.Empty:
                continue

            objects = self._run_inference(frame)
            rendered = self._render(frame, objects)
            self._drop_and_put(self._display_q, rendered)
            self._last_infer = time.time()

    # ------------------------------------------------------------------
    # Inferência YOLO + ByteTrack
    # ------------------------------------------------------------------

    def _run_inference(self, frame: np.ndarray) -> List[TrackedObject]:
        try:
            h, w = frame.shape[:2]
            if w > self._INFER_WIDTH:
                scale = self._INFER_WIDTH / w
                small = cv2.resize(frame, (int(w * scale), int(h * scale)))
            else:
                small, scale = frame, 1.0

            results = self.model.track(
                small,
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False,
                conf=0.15,
                iou=0.45,
                device=self.device,
                classes=self._VEHICLE_CLASSES,
            )

            objects: List[TrackedObject] = []
            if not results:
                return objects

            r = results[0]
            if r.boxes is None or not len(r.boxes):
                return objects

            xyxy = r.boxes.xyxy.cpu().numpy()
            cls  = r.boxes.cls.cpu().numpy().astype(int)
            conf = r.boxes.conf.cpu().numpy()
            ids  = (
                r.boxes.id.cpu().numpy().astype(int)
                if r.boxes.id is not None
                else np.full(len(xyxy), -1, dtype=int)
            )

            for i in range(len(xyxy)):
                tid = int(ids[i])
                if tid < 0:
                    continue

                x1, y1, x2, y2 = map(int, xyxy[i] / scale)
                if (x2 - x1) < 18 or (y2 - y1) < 18:
                    continue

                objects.append({
                    "track_id": tid,
                    "bbox": (x1, y1, x2, y2),
                    "center": _center_of_bbox(x1, y1, x2, y2),
                    "class_id": int(cls[i]),
                    "conf": float(conf[i]),
                })

            return objects

        except Exception as exc:
            print(f"[Inferência] Erro: {exc}")
            return []

    # ------------------------------------------------------------------
    # Renderização
    # ------------------------------------------------------------------

    def _render(self, frame: np.ndarray, objects: List[TrackedObject]) -> np.ndarray:
        out = frame.copy()
        crossed_now = self.flow.update(objects)

        # Marca quais IDs seguem ativos; remove definitivamente os que
        # ficarem `max_idle` frames seguidos sem detecção.
        active = {o["track_id"] for o in objects}
        self.flow.cleanup_stale(active)

        # Zona e linha
        ly = self.flow.line_y
        tol = self.flow.tolerance
        cv2.rectangle(out, (0, ly - tol), (out.shape[1], ly + tol), (70, 70, 70), 1)
        line_color = (0, 255, 0) if crossed_now else (0, 220, 220)
        cv2.line(out, (0, ly), (out.shape[1], ly), line_color, 3)

        # Objetos
        for obj in objects:
            x1, y1, x2, y2 = obj["bbox"]
            cx, cy = obj["center"]
            tid = obj["track_id"]

            cv2.rectangle(out, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.circle(out, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(
                out,
                f"ID:{tid} {obj['conf']:.2f}",
                (x1, max(18, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
            )

            for p, q in zip(
                self.flow.get_history(tid)[:-1],
                self.flow.get_history(tid)[1:],
            ):
                cv2.line(out, p, q, (255, 180, 0), 1)

        if crossed_now:
            cv2.putText(
                out,
                f"+{crossed_now} SUBINDO",
                (20, 260),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.1,
                (0, 255, 0),
                3,
            )

        # HUD com FPS calculado externamente
        _draw_hud(out, self.flow, self._fps, len(objects))
        return out

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _drop_and_put(q: queue.Queue, item) -> None:
        """Descarta item antigo se fila cheia, então insere o novo."""
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
    # Execução principal
    # ------------------------------------------------------------------

    def run(self) -> None:
        print("Iniciando sistema…")
        cap_t = threading.Thread(target=self._capture_loop, name="CaptureThread", daemon=True)
        proc_t = threading.Thread(target=self._process_loop, name="ProcessThread", daemon=True)
        cap_t.start()
        proc_t.start()

        print("Pressione ESC ou Q para sair.")

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

                    cv2.imshow("Fluxo Veicular", frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):  # ESC ou Q
                    break

                sleep(0.001)

        except KeyboardInterrupt:
            print("\nInterrompido pelo usuário.")
        finally:
            self.running = False
            cap_t.join(timeout=3)
            proc_t.join(timeout=3)
            self.cap.release()
            cv2.destroyAllWindows()
            with _count_lock:
                print(f"\nSessão encerrada. Total contado: {_count_up} veículos.")
