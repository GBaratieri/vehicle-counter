import os
import sys
import cv2
from video_processor import SmoothVideoProcessor

VIDEO_SOURCE = "https://camerasdaserra.com.br/stream/camerasdaserra/playlist.m3u8?ch=BZmZntmZy2GYnJu3oti2"

# O servidor de streaming só libera o playlist e os segmentos para
# requisições com Referer do próprio site (proteção contra hotlink).
# Precisa ser definido antes de qualquer cv2.VideoCapture(...).
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "headers;Referer: https://camerasdaserra.com.br/\r\n"
)


def test_video_source(video_source: str) -> bool:
    """Testa se a fonte de vídeo pode ser aberta."""
    print(f"Testando fonte de vídeo: {video_source}")
    cap = cv2.VideoCapture(video_source, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        print("Não foi possível abrir a fonte de vídeo.")
        return False

    ret, frame = cap.read()
    cap.release()

    if ret and frame is not None:
        h, w = frame.shape[:2]
        print(f"Vídeo aberto com sucesso! Resolução: {w}x{h}")
        return True

    print("Não foi possível ler frames do vídeo.")
    return False


def main():
    source = sys.argv[1] if len(sys.argv) > 1 else VIDEO_SOURCE

    if not test_video_source(source):
        print("Não foi possível abrir a câmera ao vivo. Verifique a URL e a conexão.")
        sys.exit(1)

    processor = SmoothVideoProcessor(source)
    processor.run()


if __name__ == "__main__":
    main()
