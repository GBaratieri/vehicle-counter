# Vehicle Counter

Sistema de contagem de veículos em tempo real com YOLO + ByteTrack.

## Tecnologias

| Componente | Função |
|---|---|
| YOLOv8n | Detecção de veículos (carros, ônibus, caminhões) |
| ByteTrack | Rastreamento multi-objeto com IDs persistentes |
| OpenCV | Captura e renderização de vídeo |
| Threading | Pipeline de 3 threads independentes |

## Arquitetura

```
Stream HLS
   │
   ▼
CaptureThread  ──► capture_queue ──► ProcessThread ──► display_queue ──► Main Loop (exibição)
                                           │
                                      YOLO + ByteTrack
                                      FlowCounter
```

## Arquivos

| Arquivo | Responsabilidade |
|---|---|
| `main.py` | Ponto de entrada; valida a fonte de vídeo e inicia o processador |
| `video_processor.py` | Pipeline principal: captura → YOLO/ByteTrack → exibição |
| `vehicle_tracker.py` | Rastreador euclidiano simples (fallback sem ByteTrack) |
| `vehicle_counter.py` | Processador alternativo usando `VehicleTracker` |

## Requisitos

```bash
pip install opencv-python ultralytics numpy torch torchvision
```

## Uso

```bash
# URL padrão (definida em main.py)
python main.py

# URL personalizada
python main.py https://meu-stream.com/playlist.m3u8
```

## Configurações principais

Em `video_processor.py`:

```python
# ROI: porcentagem da largura central usada
self._roi_w = int(full_w * 0.60)

# Posição vertical da linha de contagem (% da altura do ROI)
line_y=int(self._roi_h * 0.72)

# Largura máxima para inferência (velocidade × precisão)
_INFER_WIDTH = 960
```

## Performance esperada

| Dispositivo | FPS inferência |
|---|---|
| GPU (≥ 2 GB) | ~20 FPS |
| CPU | ~12 FPS |

## Controles

| Tecla | Ação |
|---|---|
| `ESC` ou `Q` | Encerrar |

## Autor

Giovanny Baratieri
