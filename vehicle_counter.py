import cv2
import numpy as np
from time import sleep
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import psutil
import time
#from constantes import *
from ultralytics import YOLO

def pega_centro(x, y, largura, altura):
    x1 = largura // 2
    y1 = altura // 2
    cx = x + x1
    cy = y + y1
    return cx, cy

class VehicleTracker:
    """Classe para rastrear veículos e detectar quando passam pela linha"""
    def __init__(self):
        self.tracked_vehicles = {}  # ID: {centro, last_y, frames_count, crossed}
        self.next_id = 0
        self.max_distance = 80  # Aumentado ainda mais para melhor rastreamento
        self.min_frames = 1     # Reduzido para 1 frame - máxima sensibilidade
        self.last_count_time = 0  # Cooldown para evitar contagem múltipla
        self.count_cooldown = 0.5  # Reduzido para 0.5 segundo - ainda mais responsivo
        self.counted_vehicle_ids = set()  # IDs que já foram contados (evita recontar o mesmo veículo)
        
    def update_vehicles(self, detections):
        """Atualizar lista de veículos rastreados"""
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
                    'last_y': old_data['current_y'],  # Usar a posição atual anterior como last_y
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
        
        # Limpar veículos antigos (não vistos por mais de 10 segundos para maior persistência)
        # Mas manter veículos que já cruzaram por mais tempo para evitar re-contagem
        self.tracked_vehicles = {}
        removed_ids = set()
        for vid, data in updated_vehicles.items():
            time_since_update = current_time - data['last_update']
            # Se já cruzou, manter por 15 segundos para evitar re-contagem
            # Se não cruzou, manter por 8 segundos
            max_time = 15.0 if data.get('crossed', False) else 8.0
            if time_since_update < max_time:
                self.tracked_vehicles[vid] = data
            else:
                removed_ids.add(vid)
        
        # Limpar IDs contados de veículos que foram removidos (após muito tempo)
        # Manter IDs por mais tempo para evitar recontar o mesmo veículo que volta
        ids_to_remove = set()
        for counted_id in self.counted_vehicle_ids:
            if counted_id in removed_ids:
                # Só remover da lista de contados após 30 segundos
                vehicle_removed_time = 30.0  # Tempo longo para garantir que não reconte
                ids_to_remove.add(counted_id)
        
        # Remover IDs muito antigos da lista de contados (limpeza periódica)
        if len(self.counted_vehicle_ids) > 50:  # Se a lista ficar muito grande
            oldest_ids = sorted(self.counted_vehicle_ids)[:10]  # Remove os 10 mais antigos
            for old_id in oldest_ids:
                if old_id not in self.tracked_vehicles:  # Só remove se não está mais sendo rastreado
                    ids_to_remove.add(old_id)
        
        self.counted_vehicle_ids -= ids_to_remove
        if ids_to_remove:
            print(f"Limpeza: removidos {len(ids_to_remove)} IDs antigos da lista de contados")
    
    def count_crossings(self, detections, line_y, offset):
        """Contar quantos veículos passaram pela linha - APENAS 1 VEZ POR ID"""
        self.update_vehicles(detections)
        
        # Verificar cooldown para evitar contagem excessiva
        current_time = time.time()
        if current_time - self.last_count_time < self.count_cooldown:
            return 0
        
        crossings = 0
        
        # Debug: mostrar informações do rastreamento
        if len(self.tracked_vehicles) > 0:
            print(f"Rastreando {len(self.tracked_vehicles)} veículo(s), linha em Y={line_y}, offset={offset}")
            print(f"IDs contados até agora: {len(self.counted_vehicle_ids)} -> {sorted(self.counted_vehicle_ids)}")
        
        for vehicle_id, vehicle_data in self.tracked_vehicles.items():
            # VERIFICAÇÃO CRUCIAL: Só contar se este ID nunca foi contado antes
            if (vehicle_data['frames_count'] >= self.min_frames and 
                not vehicle_data['crossed'] and 
                vehicle_id not in self.counted_vehicle_ids):  # <- NOVA VERIFICAÇÃO
                
                last_y = vehicle_data['last_y']
                current_y = vehicle_data['current_y']
                
                print(f"Veículo {vehicle_id}: frames={vehicle_data['frames_count']}, Y={last_y}→{current_y}, linha={line_y}")
                
                # NOVA LÓGICA: Contar quando o centro do veículo toca exatamente a linha
                # Tolerância muito pequena para contato com a linha (±1 pixel)
                line_tolerance = 1
                
                print(f"   Posições: Y anterior={last_y}, Y atual={current_y}, Linha={line_y}")
                
                # Verificar se o centro do veículo está tocando a linha (com tolerância mínima)
                center_touching_line = (line_y - line_tolerance <= current_y <= line_y + line_tolerance)
                
                # Verificar se houve movimento através da linha (para garantir que é uma passagem real)
                crossed_line = ((last_y < line_y and current_y >= line_y) or 
                               (last_y > line_y and current_y <= line_y))
                
                print(f"   Centro tocando linha? {center_touching_line} (linha±{line_tolerance}: {line_y-line_tolerance} a {line_y+line_tolerance})")
                print(f"   Cruzou a linha? {crossed_line} (movimento de {last_y} para {current_y})")
                
                # Contar apenas quando o centro toca a linha E houve movimento através dela
                if center_touching_line and crossed_line:
                    # Marcar como cruzado E adicionar à lista de contados
                    vehicle_data['crossed'] = True
                    self.counted_vehicle_ids.add(vehicle_id)  # <- ADICIONAR À LISTA DE CONTADOS
                    crossings += 1
                    direction = "↓ DESCEU" if (last_y < line_y and current_y >= line_y) else "↑ SUBIU"
                    print(f"Veículo ID:{vehicle_id} {direction} pela linha! Y: {last_y} → {current_y} (Linha: {line_y})")
                    print(f"IDs já contados: {len(self.counted_vehicle_ids)} -> {sorted(self.counted_vehicle_ids)}")
                else:
                    print(f"   Não cruzou a linha ainda")
        
        if crossings > 0:
            self.last_count_time = current_time
        
        return crossings
    
    def get_tracked_vehicles(self):
        """Retornar veículos atualmente rastreados"""
        return [(data['centro'], data['bbox'], vid, data['crossed']) 
                for vid, data in self.tracked_vehicles.items()]

def set_info(detec, frame1, linha_inicio, linha_fim, pos_linha, deslocamento_x, offset, vehicle_tracker):
    global carros, carros_lock
    
    # Contar carros que passaram pela linha (apenas rastreamento, sem backup)
    # detec já contém as detecções completas (centro, bbox)
    cars_crossed = vehicle_tracker.count_crossings(detec, pos_linha, offset)
    
    if cars_crossed > 0:
        with carros_lock:
            carros += cars_crossed
        # Desenhar linha em destaque quando carro passa
        cv2.line(frame1, (linha_inicio + deslocamento_x, pos_linha), (linha_fim + deslocamento_x, pos_linha), (0, 255, 0), 8)
        print(f"{cars_crossed} carro(s) PASSOU(RAM) pela linha! Total: {carros}")
        
        # Aviso visual na tela quando carro passa
        cv2.putText(frame1, f"CARRO PASSOU! +{cars_crossed}", (50, 250), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    else:
        # Linha normal
        cv2.line(frame1, (linha_inicio + deslocamento_x, pos_linha), (linha_fim + deslocamento_x, pos_linha), (0, 255, 255), 3)

def show_info(frame1, vehicle_tracker):
    with carros_lock:
        # Contador principal grande e destacado - SEM fundo preto
        cv2.putText(frame1, f"CARROS: {carros}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0,255,0), 5)
        
        # Mostrar informações de debug sobre IDs únicos contados
        num_unique_ids = len(vehicle_tracker.counted_vehicle_ids)
        cv2.putText(frame1, f"IDs Contados: {num_unique_ids}", (50, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
        
        # Mostrar quantos veículos estão sendo rastreados atualmente
        num_tracked = len(vehicle_tracker.tracked_vehicles)
        cv2.putText(frame1, f"Rastreados: {num_tracked}", (50, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

class ManualSmoothVideoProcessor:
    def __init__(self):
        # Configuração global para contagem
        global carros, carros_lock
        carros = 0
        carros_lock = threading.Lock()
        
        self.carros = 0
        self.carros_lock = threading.Lock()
        self.running = True
        
        # Sistema de rastreamento de veículos
        self.vehicle_tracker = VehicleTracker()
        
        # Filas ultra otimizadas
        self.capture_queue = queue.Queue(maxsize=2)  # Apenas 2 frames na fila
        self.display_queue = queue.Queue(maxsize=2)
        
        # Configuração conservadora para estabilidade
        self.num_cores = multiprocessing.cpu_count()
        self.num_workers = min(2, self.num_cores - 2)  # Máximo 2 workers
        
        # Configuração de vídeo
        self.setup_video()
        
        # Detectar e configurar dispositivo (GPU ou CPU)
        self.device = self.detect_device()
        
        # Carregar modelo YOLO otimizado
        print("Carregando modelo YOLO...")
        self.model = YOLO('yolov8n.pt')
        self.model.overrides['verbose'] = False
        self.model.overrides['device'] = self.device
        
        # Configurações específicas para GPU/CPU
        if self.device != 'cpu':
            print(f"Usando GPU: {self.device}")
            self.model.overrides['half'] = True  # Precisão half para GPU (mais rápido)
            # Fazer uma predição inicial para "aquecer" a GPU
            dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
            _ = self.model.predict(dummy_frame, verbose=False)
            print("GPU aquecida e pronta!")
        else:
            print("Usando CPU")
            self.model.overrides['half'] = False  # Precisão completa para CPU
        
        print("Modelo carregado e otimizado")
        
        # Controle de velocidade de processamento
        self.last_process_time = 0
        # Ajustar FPS baseado no dispositivo
        if self.device != 'cpu':
            self.process_interval = 1.0 / 20  # 20 FPS com GPU (mais rápido)
        else:
            self.process_interval = 1.0 / 12  # 12 FPS com CPU (estável)
        
        # Configurações de offset para detecção
        self.offset = 30  # Área maior para detecção mais permissiva da passagem
        
    def detect_device(self):
        """Detectar automaticamente o melhor dispositivo (GPU/CPU)"""
        try:
            import torch
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0)
                print(f"GPU detectada: {gpu_name}")
                print(f"{device_count} GPU(s) disponível(is)")
                
                # Verificar memória da GPU
                memory_mb = torch.cuda.get_device_properties(0).total_memory / 1024**2
                print(f"Memória GPU: {memory_mb:.0f} MB")
                
                if memory_mb > 2000:  # Pelo menos 2GB
                    return '0'  # Usar primeira GPU
                else:
                    print("GPU com pouca memória, usando CPU")
                    return 'cpu'
            else:
                print("CUDA não disponível")
                return 'cpu'
        except ImportError:
            print("PyTorch não encontrado, usando CPU")
            return 'cpu'
        except Exception as e:
            print(f"Erro ao detectar GPU: {e}")
            return 'cpu'
    
    def setup_video(self):
        video_url = "https://camerasdaserra.com.br/stream/BZmZntmZy2GYnJu3oti2/a/playlist.m3u8"
        self.cap = cv2.VideoCapture(video_url)
        
        # Configurações otimizadas para streaming ao vivo
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Buffer mínimo para menor latência
        self.cap.set(cv2.CAP_PROP_FPS, 30)  # FPS mais alto
        
        # Configurações de timeout melhoradas
        self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)  # 10s timeout para conectar
        self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)   # 5s timeout para leitura
        
        # Configurações específicas para streaming
        try:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
            # Tentar configurações adicionais para melhor streaming
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        except:
            pass
        
        ret, frame_full = self.cap.read()
        if ret and frame_full is not None:
            frame_height, frame_width = frame_full.shape[:2]
            print(f"Resolução: {frame_width}x{frame_height}")
        else:
            frame_width = 1280
            frame_height = 720
            print("Usando resolução padrão: 1280x720")
        
        self.foco_largura = int(frame_width * 0.3)
        self.foco_altura = frame_height
        self.start_x = (frame_width - self.foco_largura) // 2
        self.start_y = 0
        
        # Configurações da linha de contagem
        pixels_3cm = 55
        self.linha_inicio = pixels_3cm
        self.linha_fim = self.foco_largura - pixels_3cm
        self.pos_linha = int(self.foco_altura * 0.8)
        self.deslocamento_x = 50
        
    def capture_thread(self):
        """Thread dedicada para captura - máxima prioridade na fluidez"""
        print("Thread de captura iniciada")
        frame_count = 0
        last_frame_time = time.time()
        reconnect_count = 0
        
        while self.running:
            ret, frame_full = self.cap.read()
            current_time = time.time()
            
            if not ret or frame_full is None:
                print(f"Erro na captura (tentativa {reconnect_count + 1}), reconectando...")
                
                # Tentar reconectar
                self.cap.release()
                sleep(0.5)  # Aguardar antes de reconectar
                
                self.cap = cv2.VideoCapture("https://camerasdaserra.com.br/stream/BZmZntmZy2GYnJu3oti2/a/playlist.m3u8")
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
                self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
                
                reconnect_count += 1
                if reconnect_count > 5:
                    print("Muitas falhas de conexão, pausando...")
                    sleep(2)
                    reconnect_count = 0
                continue
            
            # Resetar contador de reconexão se conseguiu capturar
            reconnect_count = 0
            frame_count += 1
            
            # Verificar se houve "pulo" de tempo muito grande
            time_diff = current_time - last_frame_time
            if time_diff > 0.2:  # Mais de 200ms entre frames
                print(f"Possível pulo de frame detectado: {time_diff:.3f}s")
            
            last_frame_time = current_time
            
            # Recortar região de interesse
            frame1 = frame_full[self.start_y:self.start_y + self.foco_altura, 
                               self.start_x:self.start_x + self.foco_largura]
            
            # Limpar fila se cheia (priorizar frames recentes)
            while self.capture_queue.qsize() >= 2:
                try:
                    self.capture_queue.get_nowait()
                except queue.Empty:
                    break
            
            # Adicionar frame
            try:
                self.capture_queue.put_nowait(frame1.copy())
            except queue.Full:
                pass
            
            # FPS de captura mais adaptativo
            if self.device != 'cpu':
                sleep(1/25)  # 25 FPS com GPU
            else:
                sleep(1/20)  # 20 FPS com CPU
    
    def process_thread(self):
        """Thread para processamento YOLO - balanceado"""
        print("Thread de processamento iniciada")
        
        while self.running:
            current_time = time.time()
            
            # Controle de velocidade de processamento
            if current_time - self.last_process_time < self.process_interval:
                sleep(0.01)
                continue
            
            try:
                frame1 = self.capture_queue.get(timeout=0.1)
                
                # Processamento rápido
                detec = self.process_frame_fast(frame1)
                
                # Aplicar detecções ao frame
                processed_frame = self.draw_detections(frame1.copy(), detec)
                
                # Limpar fila de exibição
                while self.display_queue.qsize() >= 2:
                    try:
                        self.display_queue.get_nowait()
                    except queue.Empty:
                        break
                
                # Enviar para exibição
                try:
                    self.display_queue.put_nowait(processed_frame)
                except queue.Full:
                    pass
                
                self.last_process_time = current_time
                
            except queue.Empty:
                sleep(0.01)
    
    def process_frame_fast(self, frame1):
        """Processamento YOLO ultra otimizado com detecção melhorada"""
        try:
            # Redimensionar de forma menos agressiva para manter precisão
            height, width = frame1.shape[:2]
            if width > 480:  # Menos agressivo para melhor detecção
                scale = 480 / width
                new_width = int(width * scale)
                new_height = int(height * scale)
                frame_resized = cv2.resize(frame1, (new_width, new_height))
            else:
                frame_resized = frame1
                scale = 1.0
            
            # YOLO com configurações otimizadas para GPU/CPU
            results = self.model.predict(frame_resized, verbose=False, conf=0.15,  # Confiança reduzida
                                       device=self.device, iou=0.4, max_det=15,  # Usar dispositivo detectado
                                       augment=False, agnostic_nms=False)
            
            detec = []
            for r in results:
                if r.boxes is not None and len(r.boxes) > 0:
                    for box in r.boxes:
                        cls = int(box.cls[0])
                        if cls in [2, 5, 7]:  # carros, ônibus, caminhões
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            
                            # Reescalar coordenadas
                            if scale != 1.0:
                                x1, y1, x2, y2 = int(x1/scale), int(y1/scale), int(x2/scale), int(y2/scale)
                            
                            w, h = x2 - x1, y2 - y1
                            
                            # Filtro de tamanho MUITO mais permissivo
                            if w > 15 and h > 15 and w < 400 and h < 300:  # Muito mais permissivo
                                centro = pega_centro(x1, y1, w, h)
                                confidence = float(box.conf[0])
                                detec.append((centro, (x1, y1, x2, y2), confidence))
            
            # Debug: mostrar quantas detecções foram encontradas
            if len(detec) > 0:
                print(f"YOLO detectou {len(detec)} veículo(s)")
            
            # Filtro de proximidade mais inteligente
            filtered_detec = []
            for centro, bbox, conf in detec:
                muito_proximo = False
                for c, _, existing_conf in filtered_detec:
                    if abs(centro[0] - c[0]) < 30 and abs(centro[1] - c[1]) < 30:
                        # Manter o de maior confiança
                        if conf > existing_conf:
                            filtered_detec.remove((c, _, existing_conf))
                        else:
                            muito_proximo = True
                        break
                if not muito_proximo:
                    filtered_detec.append((centro, bbox, conf))
            
            # Ordenar por confiança e manter os melhores
            filtered_detec = sorted(filtered_detec, key=lambda x: x[2], reverse=True)
            
            # Debug: mostrar detecções filtradas
            if len(filtered_detec) > 0:
                print(f"Após filtros: {len(filtered_detec)} veículo(s) válido(s)")
            
            # Retornar apenas centro e bbox (sem confiança)
            return [(centro, bbox) for centro, bbox, _ in filtered_detec[:10]]
            
        except Exception as e:
            print(f"Erro no processamento: {e}")
            return []
    
    def draw_detections(self, frame1, detec):
        """Desenhar detecções no frame com rastreamento limpo"""
        # Obter veículos rastreados
        tracked_vehicles = self.vehicle_tracker.get_tracked_vehicles()
        
        # Debug: mostrar TODAS as detecções brutas (discretamente)
        for centro, (x1, y1, x2, y2) in detec:
            cv2.rectangle(frame1, (x1, y1), (x2, y2), (0, 255, 0), 1)  # Verde claro
            cv2.circle(frame1, centro, 2, (0, 0, 255), -1)  # Centro vermelho menor
        
        # Desenhar veículos rastreados
        for centro, bbox, vehicle_id, crossed in tracked_vehicles:
            x1, y1, x2, y2 = bbox
            
            # Cor diferente apenas para veículos que já cruzaram
            if crossed:
                color = (0, 255, 255)  # Amarelo para veículos que passaram
                cv2.rectangle(frame1, (x1, y1), (x2, y2), color, 3)
                cv2.putText(frame1, "X", (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            else:
                # Retângulo para veículos sendo rastreados
                color = (255, 0, 255)  # Magenta para rastreados
                cv2.rectangle(frame1, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame1, f"ID:{vehicle_id}", (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                # Mostrar a posição Y do centro do veículo para debug
                cv2.putText(frame1, f"Y:{centro[1]}", (x1, y2+15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        
        # Destacar melhor a linha de contagem
        linha_y = self.pos_linha
        # Desenhar zona da linha (área de detecção)
        zona_linha = 50  # Mesma zona usada na lógica
        cv2.rectangle(frame1, (0, linha_y - zona_linha), (frame1.shape[1], linha_y + zona_linha), 
                     (100, 100, 100), 1)  # Área cinza da zona
        cv2.putText(frame1, f"ZONA LINHA Y:{linha_y}", (10, linha_y - zona_linha - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Linha de contagem principal
        linha_y = self.pos_linha
        
        # Contar carros que PASSARAM pela linha
        # Passar as detecções completas (centro, bbox) para o tracker
        set_info(detec, frame1, self.linha_inicio, self.linha_fim, 
                self.pos_linha, self.deslocamento_x, self.offset, self.vehicle_tracker)
        
        # Mostrar informações
        show_info(frame1, self.vehicle_tracker)
        
        # Informações de debug (posicionadas abaixo do contador)
        cv2.putText(frame1, f"Detecções: {len(detec)}", (30, 130), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(frame1, f"Rastreados: {len(tracked_vehicles)}", (30, 155), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(frame1, f"Linha Y: {linha_y}", (30, 180), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(frame1, f"IDs Contados: {len(self.vehicle_tracker.counted_vehicle_ids)}", (30, 205), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        
        return frame1
    
    def display_thread(self):
        """Thread dedicada para exibição - máxima fluidez"""
        print("Thread de exibição iniciada")
        
        frame_count = 0
        start_time = time.time()
        
        while self.running:
            try:
                frame = self.display_queue.get(timeout=0.05)
                
                # Estatísticas
                frame_count += 1
                elapsed = time.time() - start_time
                if elapsed > 0:
                    fps = frame_count / elapsed
                    cv2.putText(frame, f"FPS: {fps:.1f}", (30, 230), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                
                cv2.imshow("Vehicle Counter - CONTAGEM POR PASSAGEM", frame)
                
                # Reset estatísticas
                if elapsed > 30:
                    frame_count = 0
                    start_time = time.time()
                
            except queue.Empty:
                pass
            
            # Verificar tecla ESC
            if cv2.waitKey(1) == 27:
                self.running = False
                break
            
            # Sleep mínimo para responsividade
            sleep(0.001)
    
    def run(self):
        """Executar sistema completo"""
        print(f"Iniciando sistema FLUIDO com {self.num_cores} cores")
        print(f"Dispositivo de processamento: {self.device}")
        print("CONFIGURAÇÃO: Captura + Processamento + Exibição separados")
        
        # Criar threads
        capture_thread = threading.Thread(target=self.capture_thread, name="CaptureThread")
        process_thread = threading.Thread(target=self.process_thread, name="ProcessThread")
        display_thread = threading.Thread(target=self.display_thread, name="DisplayThread")
        
        # Iniciar threads
        capture_thread.start()
        process_thread.start()
        display_thread.start()
        
        print("Sistema iniciado! Pressione ESC para sair")
        
        try:
            # Aguardar finalização
            display_thread.join()
            
        except KeyboardInterrupt:
            print("\nInterrompido pelo usuário")
        
        finally:
            print("Finalizando sistema...")
            self.running = False
            
            # Aguardar threads
            capture_thread.join(timeout=3)
            process_thread.join(timeout=3)
            if display_thread.is_alive():
                display_thread.join(timeout=3)
            
            # Limpar recursos
            self.cap.release()
            cv2.destroyAllWindows()
            
            print("Sistema finalizado!")

# Configuração global
carros = 0
carros_lock = threading.Lock()

if __name__ == "__main__":
    processor = ManualSmoothVideoProcessor()
    processor.run()
