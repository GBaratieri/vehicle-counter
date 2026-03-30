# Vehicle Counter

Sistema de contagem de veículos em tempo real usando YOLO e rastreamento com ByteTrack.

## Descrição

Este projeto utiliza:
- **YOLO (YOLOv8)** para detecção de veículos em tempo real
- **ByteTrack** para rastreamento de múltiplos objetos
- **OpenCV** para processamento de vídeo
- **Threading** para otimização de performance

O sistema conta veículos que passam por uma linha de detecção em um stream de vídeo ao vivo.

## Arquivos

- `main.py` - Ponto de entrada do programa
- `vehicle_counter.py` - Lógica principal de contagem e rastreamento
- `video_processor.py` - Processador de vídeo com ByteTrack
- `vehicle_tracker.py` - Classe utilitária para rastreamento
- `yolov8n.pt` - Modelo YOLO (não incluído no repositório, baixe automaticamente)

## Requisitos

```bash
pip install opencv-python ultralytics numpy torch torchvision
```

## Uso

```bash
python main.py
```

## Funcionalidades

- Detecção de veículos (carros, ônibus, caminhões)
- Rastreamento com IDs únicos
- Contagem de cruzamentos de linha
- Suporte para GPU/CPU automático
- Stream de vídeo ao vivo
- Interface visual em tempo real

## Configuração

A fonte de vídeo está configurada em `video_processor.py`:
```python
video_url = "https://camerasdaserra.com.br/stream/BZmZntmZy2GYnJu3oti2/a/playlist.m3u8"
```

## Performance

- GPU: 20 FPS
- CPU: 12 FPS

## Autor

Giovanny Baratieri
