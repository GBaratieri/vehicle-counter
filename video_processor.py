import cv2
import numpy as np
from time import sleep
import threading
import queue
import multiprocessing
import time
from collections import defaultdict
from ultralytics import YOLO

# Contadores globais
carros_subindo = 0
carros_total = 0
carros_lock = threading.Lock()


class FlowCounter:
    """
    Conta fluxo de veículos com base em track_id + cruzamento de linha.
    Considera apenas veículos SUBINDO.
    """

    def __init__(self, line_y, tolerance=12):
        self.line_y = line_y
        self.tolerance = tolerance
        self.last_centers = {}         # track_id -> (x, y)
        self.counted_up_ids = set()    # IDs já contados subindo
        self.track_history = defaultdict(list)

    def update(self, tracked_objects):
        """
        tracked_objects: lista de dicts com:
        {
            "track_id": int,
            "bbox": (x1, y1, x2, y2),
            "center": (cx, cy),
            "class_id": int,
            "conf": float
        }
        """
        global carros_subindo, carros_total, carros_lock

        crossed_up_now = 0

        for obj in tracked_objects:
            track_id = obj["track_id"]
            cx, cy = obj["center"]

            self.track_history[track_id].append((cx, cy))
            if len(self.track_history[track_id]) > 30:
                self.track_history[track_id].pop(0)

            if track_id not in self.last_centers:
                self.last_centers[track_id] = (cx, cy)
                continue

            _, prev_y = self.last_centers[track_id]

            # Contagem apenas quando o centro cruza a linha subindo
            # Verifica se cruzou de cima para baixo (prev_y > line_y e cy <= line_y)
            crossed_up = (prev_y > self.line_y and cy <= self.line_y)

            if crossed_up and track_id not in self.counted_up_ids:
                self.counted_up_ids.add(track_id)
                crossed_up_now += 1

            self.last_centers[track_id] = (cx, cy)

        if crossed_up_now > 0:
            with carros_lock:
                carros_subindo += crossed_up_now
                carros_total = carros_subindo

        return crossed_up_now

    def get_history(self, track_id):
        return self.track_history.get(track_id, [])


def pega_centro(x1, y1, x2, y2):
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    return cx, cy


def show_info(frame, flow_counter, fps_value, num_tracks):
    global carros_subindo, carros_total, carros_lock

    with carros_lock:
        cv2.putText(
            frame,
            f"FLUXO TOTAL: {carros_total}",
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            3
        )
        cv2.putText(
            frame,
            f"SUBINDO: {carros_subindo}",
            (30, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 0),
            2
        )

    cv2.putText(
        frame,
        f"TRACKS: {num_tracks}",
        (30, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"FPS: {fps_value:.1f}",
        (30, 175),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Linha Y: {flow_counter.line_y}",
        (30, 210),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


class SmoothVideoProcessor:
    """
    Processador de vídeo com YOLO + ByteTrack para fluxo veicular.
    Considera apenas veículos subindo.
    """

    def __init__(self, video_source):
        self.video_source = video_source
        self.running = True

        self.capture_queue = queue.Queue(maxsize=2)
        self.display_queue = queue.Queue(maxsize=2)

        self.num_cores = multiprocessing.cpu_count()

        self.device = self.detect_device()
        self.setup_video()

        # Linha mais baixa, mas não tão baixa a ponto de perder a contagem
        self.flow_counter = FlowCounter(
            line_y=int(self.foco_altura * 0.72),
            tolerance=12
        )

        print("Carregando modelo YOLO...")
        self.model = YOLO("yolov8n.pt")
        self.model.overrides["verbose"] = False
        self.model.overrides["device"] = self.device

        if self.device != "cpu":
            print(f"Usando GPU: {self.device}")
            self.model.overrides["half"] = True
            dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
            _ = self.model.track(
                dummy_frame,
                persist=True,
                verbose=False,
                tracker="bytetrack.yaml"
            )
            self.process_interval = 1.0 / 20
        else:
            print("Usando CPU")
            self.model.overrides["half"] = False
            self.process_interval = 1.0 / 12

        print("Modelo carregado com ByteTrack")
        self.last_process_time = 0

    def detect_device(self):
        try:
            import torch
            if torch.cuda.is_available():
                return "0"
            return "cpu"
        except ImportError:
            return "cpu"

    def setup_video(self):
        self.cap = cv2.VideoCapture(self.video_source, cv2.CAP_FFMPEG)

        if not self.cap.isOpened():
            raise RuntimeError(f"Nao foi possivel abrir a fonte de video: {self.video_source}")

        ret, frame_full = self.cap.read()
        if not ret or frame_full is None:
            raise RuntimeError("Nao foi possivel ler o primeiro frame da camera ao vivo.")

        self.frame_height, self.frame_width = frame_full.shape[:2]
        print(f"Resolucao original da camera: {self.frame_width}x{self.frame_height}")

        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        # FOCO MAIS NO CENTRO DO VÍDEO
        # Usa 60% da largura central da imagem
        self.foco_largura = int(self.frame_width * 0.60)
        self.foco_altura = self.frame_height
        self.start_x = (self.frame_width - self.foco_largura) // 2
        self.start_y = 0

        print(f"Area de foco central: {self.foco_largura}x{self.foco_altura}")

    def capture_thread(self):
        print("Thread de captura iniciada")

        while self.running:
            ret, frame_full = self.cap.read()

            if not ret or frame_full is None:
                print("Falha ao ler frame da live, tentando continuar...")
                sleep(0.05)
                continue

            frame = frame_full[
                self.start_y:self.start_y + self.foco_altura,
                self.start_x:self.start_x + self.foco_largura
            ]

            while self.capture_queue.qsize() >= 2:
                try:
                    self.capture_queue.get_nowait()
                except queue.Empty:
                    break

            try:
                self.capture_queue.put_nowait(frame.copy())
            except queue.Full:
                pass

            if self.device != "cpu":
                sleep(1 / 30)
            else:
                sleep(1 / 20)

    def process_thread(self):
        print("Thread de processamento iniciada")

        while self.running:
            current_time = time.time()

            if current_time - self.last_process_time < self.process_interval:
                sleep(0.01)
                continue

            try:
                frame = self.capture_queue.get(timeout=0.1)

                tracked_objects = self.process_frame_track(frame)
                processed_frame = self.draw_detections(frame.copy(), tracked_objects)

                while self.display_queue.qsize() >= 2:
                    try:
                        self.display_queue.get_nowait()
                    except queue.Empty:
                        break

                try:
                    self.display_queue.put_nowait(processed_frame)
                except queue.Full:
                    pass

                self.last_process_time = current_time

            except queue.Empty:
                sleep(0.01)

    def process_frame_track(self, frame):
        """
        Usa YOLO + ByteTrack.
        """
        try:
            height, width = frame.shape[:2]

            if width > 960:
                scale = 960 / width
                new_width = int(width * scale)
                new_height = int(height * scale)
                frame_resized = cv2.resize(frame, (new_width, new_height))
            else:
                frame_resized = frame
                scale = 1.0

            results = self.model.track(
                frame_resized,
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False,
                conf=0.15,
                iou=0.45,
                device=self.device,
                classes=[2, 5, 7]
            )

            tracked_objects = []

            if not results:
                return tracked_objects

            result = results[0]
            if result.boxes is None or len(result.boxes) == 0:
                return tracked_objects

            boxes = result.boxes

            xyxy = boxes.xyxy.cpu().numpy() if boxes.xyxy is not None else []
            cls = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else []
            confs = boxes.conf.cpu().numpy() if boxes.conf is not None else []

            if boxes.id is not None:
                ids = boxes.id.cpu().numpy().astype(int)
            else:
                ids = [-1] * len(xyxy)

            for i in range(len(xyxy)):
                track_id = int(ids[i])
                if track_id < 0:
                    continue

                x1, y1, x2, y2 = map(int, xyxy[i])

                if scale != 1.0:
                    x1 = int(x1 / scale)
                    y1 = int(y1 / scale)
                    x2 = int(x2 / scale)
                    y2 = int(y2 / scale)

                w = x2 - x1
                h = y2 - y1

                if w < 18 or h < 18:
                    continue

                center = pega_centro(x1, y1, x2, y2)

                tracked_objects.append({
                    "track_id": track_id,
                    "bbox": (x1, y1, x2, y2),
                    "center": center,
                    "class_id": int(cls[i]),
                    "conf": float(confs[i])
                })

            return tracked_objects

        except Exception as e:
            print(f"Erro no tracking: {e}")
            return []

    def draw_detections(self, frame, tracked_objects):
        crossed_up_now = self.flow_counter.update(tracked_objects)

        line_y = self.flow_counter.line_y
        tolerance = self.flow_counter.tolerance

        # zona da linha
        cv2.rectangle(
            frame,
            (0, line_y - tolerance),
            (frame.shape[1], line_y + tolerance),
            (80, 80, 80),
            1
        )

        # linha principal
        line_color = (0, 255, 0) if crossed_up_now > 0 else (0, 255, 255)
        cv2.line(frame, (0, line_y), (frame.shape[1], line_y), line_color, 3)

        for obj in tracked_objects:
            x1, y1, x2, y2 = obj["bbox"]
            cx, cy = obj["center"]
            track_id = obj["track_id"]
            conf = obj["conf"]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)

            cv2.putText(
                frame,
                f"ID:{track_id} {conf:.2f}",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )

            history = self.flow_counter.get_history(track_id)
            for i in range(1, len(history)):
                cv2.line(frame, history[i - 1], history[i], (255, 200, 0), 2)

        if crossed_up_now > 0:
            cv2.putText(
                frame,
                f"SUBINDO +{crossed_up_now}",
                (30, 250),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 0),
                3
            )

        return frame

    def run(self):
        print(f"Iniciando sistema com {self.num_cores} cores")
        print(f"Dispositivo: {self.device}")
        print(f"Fonte ao vivo: {self.video_source}")

        capture_thread = threading.Thread(target=self.capture_thread, name="CaptureThread")
        process_thread = threading.Thread(target=self.process_thread, name="ProcessThread")

        capture_thread.start()
        process_thread.start()

        print("Sistema iniciado! Pressione ESC para sair")

        frame_count = 0
        start_time = time.time()

        try:
            while self.running:
                try:
                    frame = self.display_queue.get(timeout=0.05)

                    frame_count += 1
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed if elapsed > 0 else 0.0

                    show_info(frame, self.flow_counter, fps, 0 if frame is None else 0)

                    cv2.imshow("Fluxo Veicular - SUBINDO", frame)

                    if elapsed > 30:
                        frame_count = 0
                        start_time = time.time()

                except queue.Empty:
                    pass

                if cv2.waitKey(1) & 0xFF == 27:
                    self.running = False
                    break

                sleep(0.001)

        except KeyboardInterrupt:
            print("\nInterrompido")

        finally:
            print("Finalizando...")
            self.running = False

            capture_thread.join(timeout=3)
            process_thread.join(timeout=3)

            self.cap.release()
            cv2.destroyAllWindows()
            print("Finalizado!")