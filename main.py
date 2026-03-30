

import cv2
from video_processor import SmoothVideoProcessor

def test_video_source(video_source):
    """
    Testa se a fonte de vídeo pode ser aberta.
    """
    print(f"Testando fonte de vídeo: {video_source}")
    cap = cv2.VideoCapture(video_source, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        print("Nao foi possivel abrir a fonte de video.")
        return False

    ret, frame = cap.read()
    if ret and frame is not None:
        height, width = frame.shape[:2]
        print(f"Video aberto com sucesso! Resolucao: {width}x{height}")
        cap.release()
        return True
    else:
        print("Nao foi possivel ler frames do video.")
        cap.release()
        return False

def main():
    video_source = "https://tv.rbt.psi.br/media/n/BZmZntmZy2GYnJu3oti2/a/playlist.m3u8"

    if not test_video_source(video_source):
        print("Nao foi possivel abrir a camera ao vivo.")
        return

    processor = SmoothVideoProcessor(video_source)
    processor.run()

if __name__ == "__main__":
    main()