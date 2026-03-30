import cv2
import numpy as np
from time import sleep
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import psutil
import time
from ultralytics import YOLO

def pega_centro(x, y, largura, altura):
    """
    Calcula o centro de um retângulo.
    """
    x1 = largura // 2
    y1 = altura // 2
    cx = x + x1
    cy = y + y1
    return cx, cy

class VehicleTracker:
    """
    Classe para rastrear veículos e detectar quando passam pela linha.
    """
    def __init__(self):
        self.tracked_vehicles = {}  # ID: {centro, last_y, frames_count, crossed}
        self.next_id = 0
        self.max_distance = 80  # Distância máxima para associar veículos
        self.min_frames = 1     # Mínimo de frames para considerar válido
        self.last_count_time = 0  # Cooldown para evitar contagem múltipla
        self.count_cooldown = 0.5  # Tempo de cooldown em segundos
        self.counted_vehicle_ids = set()  # IDs que já foram contados

    def update_vehicles(self, detections):
        """
        Atualiza a lista de veículos rastreados com novas detecções.
        """
        current_time = time.time()
        updated_vehicles = {}

        for centro, bbox in detections:
            x, y = centro
            best_match_id = None
            best_distance = float('inf')

            # Procurar veículo existente mais próximo
            for vehicle_id, vehicle_data in self.tracked_vehicles.items():
                old_x, old_y = vehicle_data['centro']
                distance = ((x - old_x)**2 + (y - old_y)**2)**0.5

                if distance < self.max_distance and distance < best_distance:
                    best_distance = distance
                    best_match_id = vehicle_id

            if best_match_id is not None:
                # Atualizar veículo existente
                old_data = self.tracked_vehicles[best_match_id]
                updated_vehicles[best_match_id] = {
                    'centro': (x, y),
                    'last_y': old_data['current_y'],
                    'current_y': y,
                    'frames_count': old_data['frames_count'] + 1,
                    'crossed': old_data.get('crossed', False),
                    'bbox': bbox,
                    'last_update': current_time
                }
            else:
                # Novo veículo
                updated_vehicles[self.next_id] = {
                    'centro': (x, y),
                    'last_y': y,
                    'current_y': y,
                    'frames_count': 1,
                    'crossed': False,
                    'bbox': bbox,
                    'last_update': current_time
                }
                self.next_id += 1

        # Limpar veículos antigos
        self.tracked_vehicles = {}
        removed_ids = set()
        for vid, data in updated_vehicles.items():
            time_since_update = current_time - data['last_update']
            max_time = 15.0 if data.get('crossed', False) else 8.0
            if time_since_update < max_time:
                self.tracked_vehicles[vid] = data
            else:
                removed_ids.add(vid)

        # Limpar IDs contados de veículos removidos
        ids_to_remove = set()
        for counted_id in self.counted_vehicle_ids:
            if counted_id in removed_ids:
                ids_to_remove.add(counted_id)

        self.counted_vehicle_ids -= ids_to_remove
        if ids_to_remove:
            print(f"🧹 Limpeza: removidos {len(ids_to_remove)} IDs antigos")

    def count_crossings(self, detections, line_y, offset):
        """
        Conta quantos veículos passaram pela linha.
        """
        self.update_vehicles(detections)

        current_time = time.time()
        # if current_time - self.last_count_time < self.count_cooldown:
        #     return 0

        crossings = 0

        for vehicle_id, vehicle_data in self.tracked_vehicles.items():
            if (vehicle_data['frames_count'] >= self.min_frames and
                not vehicle_data['crossed'] and
                vehicle_id not in self.counted_vehicle_ids):

                last_y = vehicle_data['last_y']
                current_y = vehicle_data['current_y']

                crossed_line = ((last_y < line_y and current_y >= line_y) or
                               (last_y > line_y and current_y <= line_y))

                if crossed_line:
                    vehicle_data['crossed'] = True
                    self.counted_vehicle_ids.add(vehicle_id)
                    crossings += 1
                    direction = "↓ DESCEU" if (last_y < line_y and current_y >= line_y) else "↑ SUBIU"
                    print(f"Veículo ID:{vehicle_id} {direction} pela linha!")

        if crossings > 0:
            self.last_count_time = current_time

        return crossings

    def get_tracked_vehicles(self):
        """
        Retorna veículos atualmente rastreados.
        """
        return [(data['centro'], data['bbox'], vid, data['crossed'])
                for vid, data in self.tracked_vehicles.items()]