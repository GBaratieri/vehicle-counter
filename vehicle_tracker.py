"""
vehicle_tracker.py
------------------
Rastreador simples baseado em distância euclidiana.
Usado como fallback quando o ByteTrack não está disponível.
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

Detection = Tuple[Tuple[int, int], Tuple[int, int, int, int]]  # (center, bbox)


@dataclass
class TrackedVehicle:
    center: Tuple[int, int]
    bbox: Tuple[int, int, int, int]
    last_y: int
    current_y: int
    frames_count: int = 1
    crossed: bool = False
    last_update: float = field(default_factory=time.time)


class VehicleTracker:
    """
    Rastreia veículos quadro a quadro usando distância euclidiana
    e detecta cruzamentos de linha.
    """

    MAX_DISTANCE = 80       # px – máxima distância para associar detecções
    MIN_FRAMES = 1          # frames mínimos antes de contar
    TTL_NORMAL = 8.0        # segundos para manter veículo que não cruzou
    TTL_CROSSED = 15.0      # segundos para manter veículo que já cruzou
    MAX_COUNTED_IDS = 500   # limite de IDs na memória

    def __init__(self) -> None:
        self._vehicles: Dict[int, TrackedVehicle] = {}
        self._counted_ids: Set[int] = set()
        self._next_id: int = 0
        self._last_cross_time: float = 0.0

    # ------------------------------------------------------------------
    # Atualização
    # ------------------------------------------------------------------

    def update(self, detections: List[Detection]) -> None:
        """Associa detecções aos veículos já rastreados (greedy nearest)."""
        now = time.time()
        used_ids: Set[int] = set()
        new_vehicles: Dict[int, TrackedVehicle] = {}

        for center, bbox in detections:
            cx, cy = center
            best_id, best_dist = self._nearest(cx, cy, used_ids)

            if best_id is not None:
                v = self._vehicles[best_id]
                new_vehicles[best_id] = TrackedVehicle(
                    center=(cx, cy),
                    bbox=bbox,
                    last_y=v.current_y,
                    current_y=cy,
                    frames_count=v.frames_count + 1,
                    crossed=v.crossed,
                    last_update=now,
                )
                used_ids.add(best_id)
            else:
                vid = self._next_id
                self._next_id += 1
                new_vehicles[vid] = TrackedVehicle(
                    center=(cx, cy),
                    bbox=bbox,
                    last_y=cy,
                    current_y=cy,
                    last_update=now,
                )

        # Mantém veículos não detectados neste frame até expirar o TTL, em
        # vez de descartá-los de imediato — sem isso, qualquer falha pontual
        # de detecção já "esquecia" o veículo e o cruzamento podia ser
        # perdido ou contado em duplicidade com um novo ID.
        expired: Set[int] = set()
        for vid, v in self._vehicles.items():
            if vid in used_ids:
                continue
            ttl = self.TTL_CROSSED if v.crossed else self.TTL_NORMAL
            if now - v.last_update > ttl:
                expired.add(vid)
            else:
                new_vehicles[vid] = v

        self._vehicles = new_vehicles

        # Limpa IDs contados de veículos expirados
        self._counted_ids -= expired
        if len(self._counted_ids) > self.MAX_COUNTED_IDS:
            excess = sorted(self._counted_ids)[: len(self._counted_ids) - self.MAX_COUNTED_IDS]
            self._counted_ids -= set(excess)

    def _nearest(
        self, cx: int, cy: int, used_ids: Set[int]
    ) -> Tuple[int | None, float]:
        best_id, best_dist = None, float("inf")
        for vid, v in self._vehicles.items():
            if vid in used_ids:
                continue
            ox, oy = v.center
            dist = ((cx - ox) ** 2 + (cy - oy) ** 2) ** 0.5
            if dist < self.MAX_DISTANCE and dist < best_dist:
                best_dist = dist
                best_id = vid
        return best_id, best_dist

    # ------------------------------------------------------------------
    # Contagem
    # ------------------------------------------------------------------

    def count_crossings(self, detections: List[Detection], line_y: int) -> int:
        """Atualiza rastreamento e retorna número de cruzamentos novos."""
        self.update(detections)

        crossings = 0
        for vid, v in self._vehicles.items():
            if v.frames_count < self.MIN_FRAMES:
                continue
            if v.crossed or vid in self._counted_ids:
                continue

            crossed = (v.last_y < line_y <= v.current_y) or (
                v.last_y > line_y >= v.current_y
            )
            if crossed:
                v.crossed = True
                self._counted_ids.add(vid)
                crossings += 1
                direction = "↓ DESCEU" if v.last_y < line_y else "↑ SUBIU"
                print(f"[Tracker] Veículo ID:{vid} {direction} | Y: {v.last_y}→{v.current_y}")

        if crossings:
            self._last_cross_time = time.time()

        return crossings

    # ------------------------------------------------------------------
    # Consulta
    # ------------------------------------------------------------------

    def get_tracked_vehicles(
        self,
    ) -> List[Tuple[Tuple[int, int], Tuple[int, int, int, int], int, bool]]:
        return [
            (v.center, v.bbox, vid, v.crossed)
            for vid, v in self._vehicles.items()
        ]

    @property
    def count(self) -> int:
        return len(self._counted_ids)
