import cv2
import os
from ultralytics import YOLO
import numpy as np
import torch
import time
from datetime import datetime


class HybridCardDetector:
    """
    Detecta cartas TCG com YOLOv8-OBB, usa análise de blocos para robustez à iluminação,
    alinha com warpPerspective e salva imagens de forma estável.
    Projetado para ser usado frame a frame em aplicações como Streamlit.
    """
    # Estados
    WAITING = "AGUARDANDO"
    MOTION = "MOVIMENTO"
    STABILIZING = "ESTABILIZANDO"

    def __init__(self, model_path, output_dir='cards_detected_hybrid', conf_threshold=0.4,
                 crop_width=480, crop_height=670, # Dimensões de recorte
                 grid_size=3, illumination_threshold=0.08, new_card_threshold=0.15, variance_threshold=0.015,
                 required_stability_frames=5, save_cooldown=1.0):
        """
        Inicializa o detector híbrido. Carrega o modelo e define configurações fixas.
        """
        print("--- Inicializando Detector Híbrido (YOLOv8-OBB + Análise de Blocos - Refatorado) ---")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo não encontrado: {model_path}")

        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        print(f"Carregando modelo OBB: {model_path}")
        try:
            self.model = YOLO(model_path)
            if torch.cuda.is_available():
                self.device = 'cuda'
                self.model.to(self.device)
                print(f"Usando GPU: {torch.cuda.get_device_name(0)}")
            else:
                self.device = 'cpu'
                print("Usando CPU.")
            print("Modelo carregado com sucesso.")
        except Exception as e:
            raise RuntimeError(f"Falha ao carregar o modelo YOLO: {e}")

        # Configurações YOLO e Recorte
        self.conf_threshold = conf_threshold
        self.crop_width = crop_width
        self.crop_height = crop_height

        # Configurações da Análise de Blocos
        self.grid_size = grid_size
        self.illumination_change_threshold = illumination_threshold
        self.new_card_threshold = new_card_threshold
        self.variance_threshold = variance_threshold

        # Configurações da Máquina de Estados
        self.required_stability_frames = required_stability_frames
        self.save_cooldown = save_cooldown

        # Timestamp da sessão para nomes de arquivo
        self.session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        print(f"Detector inicializado. Salvando em: {self.output_dir}")
        print(f"Análise de blocos {grid_size}x{grid_size} ATIVA.")
        print(f"Estabilidade requerida: {self.required_stability_frames} frames.")
        print(f"Cooldown salvamento: {self.save_cooldown}s.")
        print("-" * 50)

    @staticmethod
    def initialize_state():
        """Retorna o dicionário de estado inicial."""
        return {
            'state': HybridCardDetector.WAITING,
            'cards_saved_count': 0,
            'stability_counter': 0,
            'current_card_aligned_box': None,
            'last_stable_corners': None,
            'last_save_time': 0,
            'roi_defined': False,
            'roi': None,
            'previous_roi_stats': None,
            'difference_heatmap': None,
            'block_change_type': 'INICIAL' # Tipo de mudança inicial
        }

    def detect_card_yolo(self, frame):
        """Detecta carta com YOLOv8-OBB."""
        results = self.model.predict(frame, conf=self.conf_threshold, device=self.device, verbose=False)
        detections = results[0]
        aligned_box, corners, best_score = None, None, 0
        if detections.obb is not None and len(detections.obb) > 0:
            best_idx = detections.obb.conf.argmax()
            confidence = detections.obb.conf[best_idx].item()
            if confidence >= self.conf_threshold:
                best_score = confidence
                corners = detections.obb.xyxyxyxy[best_idx].cpu().numpy().reshape((4, 2))
                x_coords, y_coords = corners[:, 0], corners[:, 1]
                x1, y1, x2, y2 = np.min(x_coords), np.min(y_coords), np.max(x_coords), np.max(y_coords)
                aligned_box = [int(x1), int(y1), int(x2), int(y2)]
        return aligned_box, corners, best_score

    def is_same_position(self, box1, box2, tolerance=30):
        """
        Verifica se duas bounding boxes *alinhadas* estão aproximadamente na mesma posição.
        """
        if box1 is None or box2 is None:
            return False
        # Calcula centro da box1
        center1_x = (box1[0] + box1[2]) / 2
        center1_y = (box1[1] + box1[3]) / 2
        # Calcula centro da box2
        center2_x = (box2[0] + box2[2]) / 2
        center2_y = (box2[1] + box2[3]) / 2
        # Calcula distância Euclidiana
        distance = np.sqrt((center1_x - center2_x)**2 + (center1_y - center2_y)**2)
        # Retorna True se a distância for menor que a tolerância
        return distance < tolerance

    def define_roi(self, frame, use_detection=True):
        """Define ROI inicial. Retorna roi e roi_defined."""
        if use_detection:
            aligned_box, _, score = self.detect_card_yolo(frame)
            if aligned_box and score > self.conf_threshold:
                print(f"Carta detectada ({score:.2f}) para definir ROI inicial.")
                x1, y1, x2, y2 = aligned_box
                w, h = x2 - x1, y2 - y1
                margin_x, margin_y = int(w * 0.15), int(h * 0.15)
                roi_x = max(0, x1 - margin_x)
                roi_y = max(0, y1 - margin_y)
                roi_w = min(frame.shape[1] - roi_x, w + 2 * margin_x)
                roi_h = min(frame.shape[0] - roi_y, h + 2 * margin_y)
                roi = (roi_x, roi_y, roi_w, roi_h)
                roi_defined = True
                print(f"ROI definida em: {roi}")
                return roi, roi_defined
            else:
                print("Carta não detectada para ROI automático. Usando centro.")
        h_fr, w_fr = frame.shape[:2]
        roi_w_def, roi_h_def = int(w_fr * 0.7), int(h_fr * 0.7)
        roi = (w_fr//2 - roi_w_def//2, h_fr//2 - roi_h_def//2, roi_w_def, roi_h_def)
        roi_defined = True
        print(f"ROI padrão definida em: {roi}")
        return roi, roi_defined

    def extract_roi(self, frame, current_roi, is_roi_defined):
        """Extrai a região de interesse do frame. Retorna a ROI e pode redefinir."""
        if not is_roi_defined or current_roi is None:
            print("[AVISO] ROI não definida ou inválida, tentando definir...")
            # Nota: Define ROI com detecção desligada se falhar a primeira vez.
            # Idealmente, a lógica de quando chamar define_roi seria externa.
            new_roi, new_roi_defined = self.define_roi(frame, use_detection=False)
            if not new_roi_defined:
                print("[ERRO] Falha ao definir ROI.")
                return None, new_roi, new_roi_defined # Retorna None para imagem ROI

            current_roi = new_roi # Usa a ROI recém definida
            is_roi_defined = new_roi_defined

        x, y, w, h = current_roi
        # Garante que as coordenadas e dimensões sejam válidas
        x, y = max(0, x), max(0, y)
        y_end = min(y + h, frame.shape[0])
        x_end = min(x + w, frame.shape[1])

        # Verifica se a ROI resultante tem tamanho válido
        if y_end <= y or x_end <= x:
             print(f"[AVISO] ROI inválida após ajuste: ({x},{y},{w},{h}) -> ({x},{y}) a ({x_end},{y_end}) no frame {frame.shape[:2]}. ROI será redefinida.")
             # Retorna None e sinaliza para redefinir ROI externamente
             return None, None, False

        return frame[y:y_end, x:x_end], current_roi, is_roi_defined

    def get_block_stats(self, image):
        """Divide a imagem ROI em blocos e extrai estatísticas."""
        h, w = image.shape[:2]
        if h < self.grid_size or w < self.grid_size:
            mean_rgb = [np.mean(image[:,:,i]) for i in range(3)] if h>0 and w>0 else [0,0,0]
            std_rgb = [np.std(image[:,:,i]) for i in range(3)] if h>0 and w>0 else [0,0,0]
            return [{'position': (0, 0), 'mean': mean_rgb, 'std': std_rgb}] * (self.grid_size**2)
        block_h, block_w = h // self.grid_size, w // self.grid_size
        block_stats = []
        for y_grid in range(self.grid_size):
            for x_grid in range(self.grid_size):
                start_y, start_x = y_grid * block_h, x_grid * block_w
                end_y, end_x = min(start_y + block_h, h), min(start_x + block_w, w)
                block = image[start_y:end_y, start_x:end_x]
                if block.size == 0: continue
                mean_rgb = [np.mean(block[:,:,i]) for i in range(3)]
                std_rgb = [np.std(block[:,:,i]) for i in range(3)]
                block_stats.append({'position': (x_grid, y_grid), 'mean': mean_rgb, 'std': std_rgb})
        while len(block_stats) < self.grid_size**2:
             if block_stats: block_stats.append(block_stats[-1])
             else: block_stats.append({'position': (0, 0), 'mean': [0,0,0], 'std': [0,0,0]})
        return block_stats[:self.grid_size**2]

    def calculate_block_differences(self, current_stats, previous_stats):
        """Calcula diferenças normalizadas entre blocos."""
        if not current_stats or not previous_stats or len(current_stats) != len(previous_stats):
             return {'differences': [], 'avg_diff': 0, 'max_diff': 0, 'variance': 0, 'difference_matrix': np.zeros((self.grid_size, self.grid_size))}
        differences = []
        difference_matrix = np.zeros((self.grid_size, self.grid_size))
        current_global_mean = [np.mean([b['mean'][i] for b in current_stats if b]) for i in range(3)]
        previous_global_mean = [np.mean([b['mean'][i] for b in previous_stats if b]) for i in range(3)]
        for i, (curr, prev) in enumerate(zip(current_stats, previous_stats)):
            if not curr or not prev: continue
            block_diff = 0
            for c in range(3):
                curr_norm = curr['mean'][c] / (current_global_mean[c] + 1e-6)
                prev_norm = prev['mean'][c] / (previous_global_mean[c] + 1e-6)
                channel_diff = abs(curr_norm - prev_norm)
                max_std = max(curr['std'][c], prev['std'][c])
                std_diff = abs(curr['std'][c] - prev['std'][c]) / (max_std + 1e-6)
                block_diff += 0.7 * channel_diff + 0.3 * std_diff
            block_diff /= 3
            differences.append(block_diff)
            x, y = curr['position']
            if 0 <= y < self.grid_size and 0 <= x < self.grid_size:
                 difference_matrix[y, x] = block_diff
        avg_diff = np.mean(differences) if differences else 0
        max_diff = np.max(differences) if differences else 0
        variance = np.var(differences) if differences else 0
        return {'differences': differences, 'avg_diff': avg_diff, 'max_diff': max_diff, 'variance': variance, 'difference_matrix': difference_matrix}

    def create_difference_heatmap(self, difference_matrix):
        """Cria mapa de calor."""
        if difference_matrix is None: return None
        try:
            normalized = cv2.normalize(difference_matrix, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
            heatmap = cv2.resize(heatmap, (150, 150), interpolation=cv2.INTER_NEAREST)
            return heatmap
        except Exception: return None

    def classify_change_type(self, diff_results):
        """Classifica a mudança detectada pelos blocos."""
        avg_diff = diff_results['avg_diff']
        variance = diff_results['variance']
        score = 0.6 * avg_diff + 0.4 * np.sqrt(variance)
        if score < self.illumination_change_threshold:
            return "NO_CHANGE_OR_LIGHTING"
        elif score >= self.new_card_threshold and variance >= self.variance_threshold:
            return "NEW_CARD_MOTION"
        elif (self.illumination_change_threshold <= score < self.new_card_threshold) or \
             (score >= self.new_card_threshold and variance < self.variance_threshold):
            return "MINOR_MOTION_OR_LIGHTING"
        else:
            return "UNCERTAIN"

    def _sort_corners(self, corners):
        if corners is None or corners.shape != (4, 2): return None
        corners = corners[np.argsort(corners[:, 1]), :]
        top_corners = corners[:2, :][np.argsort(corners[:2, 0]), :]
        bottom_corners = corners[2:, :][np.argsort(corners[2:, 0]), :]
        return np.array([top_corners[0], top_corners[1], bottom_corners[1], bottom_corners[0]], dtype=np.float32)

    def _crop_and_warp(self, frame, corners):
        sorted_corners = self._sort_corners(corners)
        if sorted_corners is None: return None
        dst_points = np.array([[0, 0], [self.crop_width - 1, 0], [self.crop_width - 1, self.crop_height - 1], [0, self.crop_height - 1]], dtype=np.float32)
        try:
            matrix = cv2.getPerspectiveTransform(sorted_corners, dst_points)
            warped_image = cv2.warpPerspective(frame, matrix, (self.crop_width, self.crop_height))
            return warped_image
        except Exception as e: return None

    def save_card_with_metadata(self, frame, corners, cards_saved_count):
        """Salva a carta recortada e alinhada. Retorna o caminho e a contagem atualizada."""
        # Importante: fazer uma cópia do frame original antes de qualquer desenho
        # para garantir que a borda verde não apareça no recorte
        clean_frame = frame.copy()
        
        # Usar o frame limpo para o recorte, sem nenhuma visualização
        cropped_warped_image = self._crop_and_warp(clean_frame, corners)
        if cropped_warped_image is None: return None, cards_saved_count
        if cropped_warped_image.shape[0] < 20 or cropped_warped_image.shape[1] < 20: return None, cards_saved_count

        new_cards_saved_count = cards_saved_count + 1
        card_id = f"card_{new_cards_saved_count}"
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S_%f")
        filename = f"{card_id}_{self.session_timestamp}_{timestamp}.jpg"
        filepath = os.path.join(self.output_dir, filename)
        try:
            cv2.imwrite(filepath, cropped_warped_image)
            print(f"Carta {card_id} salva: {filepath}")
            return filepath, new_cards_saved_count # Retorna caminho e nova contagem
        except Exception as e:
            print(f"[ERRO] Falha ao salvar imagem {filepath}: {e}")
            return None, cards_saved_count # Retorna None e contagem original

    def enhance_visualization(self, frame, yolo_corners, current_state):
        """Cria visualização no frame com base no estado atual."""
        vis_frame = frame.copy()
        h, w = vis_frame.shape[:2]
        state = current_state['state']
        stability_count = current_state['stability_counter']
        cards_saved_count = current_state['cards_saved_count']
        block_change_type = current_state.get('block_change_type', 'N/A') # Pega do estado
        roi = current_state['roi']
        roi_defined = current_state['roi_defined']
        difference_heatmap = current_state['difference_heatmap']

        state_color = { self.WAITING: (255, 255, 255), self.MOTION: (0, 165, 255), self.STABILIZING: (0, 255, 255) }
        if yolo_corners is not None:
             cv2.polylines(vis_frame, [yolo_corners.astype(int)], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.putText(vis_frame, f"Estado: {state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_color.get(state, (255,0,0)), 2)
        cv2.putText(vis_frame, f"Cartas Salvas: {cards_saved_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis_frame, f"Tipo Mudanca (Bloco): {block_change_type}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)
        if state == self.STABILIZING:
             stability_text = f"Estabilizando: {stability_count}/{self.required_stability_frames}"
             cv2.putText(vis_frame, stability_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_color[self.STABILIZING], 2)
        if roi_defined and roi:
            x, y, w_roi, h_roi = roi
            # Certifica que o retângulo está dentro dos limites
            x, y = max(0, x), max(0, y)
            x2, y2 = min(x + w_roi, w), min(y + h_roi, h)
            if x2 > x and y2 > y:
                 cv2.rectangle(vis_frame, (x, y), (x2, y2), (255, 0, 0), 1)

        if difference_heatmap is not None:
            try:
                h_map, w_map = difference_heatmap.shape[:2]
                roi_vis_y_start, roi_vis_x_start = 20, vis_frame.shape[1] - w_map - 20
                if roi_vis_y_start + h_map <= vis_frame.shape[0] and roi_vis_x_start >= 0:
                     vis_frame[roi_vis_y_start : roi_vis_y_start + h_map, roi_vis_x_start : roi_vis_x_start + w_map] = difference_heatmap
                     cv2.putText(vis_frame, "Mapa Diferencas (ROI)", (roi_vis_x_start, roi_vis_y_start - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            except Exception as e:
                print(f"[AVISO] Erro ao desenhar heatmap: {e}")
        # Removido texto 'pressione q' - controle será externo
        # cv2.putText(vis_frame, "Pressione 'q' para sair", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return vis_frame

    def process_frame(self, frame, prev_frame, current_state):
        """
        Processa um único frame.

        Args:
            frame: O frame atual da câmera (numpy array BGR).
            prev_frame: O frame anterior da câmera (numpy array BGR).
            current_state: Dicionário contendo o estado atual da detecção.

        Returns:
            tuple: (
                processed_frame: O frame com visualizações (numpy array BGR).
                updated_state: O dicionário de estado atualizado para o próximo frame.
                saved_filepath: O caminho do arquivo salvo, ou None se nada foi salvo.
            )
        """
        updated_state = current_state.copy() # Começa com o estado atual
        saved_filepath = None

        # 0. Lidar com primeiro frame ou reset de ROI
        if prev_frame is None or not updated_state['roi_defined']:
            print("Primeiro frame ou ROI não definida, inicializando ROI...")
            roi, roi_defined = self.define_roi(frame, use_detection=True)
            updated_state['roi'] = roi
            updated_state['roi_defined'] = roi_defined
            updated_state['previous_roi_stats'] = None # Reseta stats do frame anterior
            updated_state['block_change_type'] = 'INICIAL'
            # Retorna frame original se não puder definir ROI
            if not roi_defined:
                print("[ERRO] Não foi possível definir ROI inicial.")
                processed_frame = self.enhance_visualization(frame, None, updated_state)
                return processed_frame, updated_state, None


        # 1. Detecção YOLO OBB
        aligned_box, corners, score = self.detect_card_yolo(frame)
        
        # Diagnóstico de score
        if corners is not None:
            print(f"Detecção YOLO: score={score:.2f}, atual={updated_state['state']}, estab={updated_state['stability_counter']}")

        # 2. Análise de Blocos (na ROI)
        # Extrai ROI atual e anterior (se possível)
        current_roi_img, roi_updated, roi_defined_updated = self.extract_roi(frame, updated_state['roi'], updated_state['roi_defined'])
        updated_state['roi'] = roi_updated
        updated_state['roi_defined'] = roi_defined_updated

        previous_roi_img = None
        if prev_frame is not None and updated_state['roi_defined']:
            # Usa a mesma ROI para o frame anterior, mesmo que tenha sido atualizada agora
            previous_roi_img, _, _ = self.extract_roi(prev_frame, updated_state['roi'], updated_state['roi_defined'])

        block_change_type = "N/A" # Default se a análise falhar
        current_stats = None
        difference_matrix = None

        if current_roi_img is not None:
             current_stats = self.get_block_stats(current_roi_img)

             if previous_roi_img is not None and updated_state['previous_roi_stats'] is not None and current_roi_img.shape == previous_roi_img.shape:
                 # Só calcula diferença se tiver stats anteriores válidos e ROIs de mesmo tamanho
                 block_diff_results = self.calculate_block_differences(current_stats, updated_state['previous_roi_stats'])
                 block_change_type = self.classify_change_type(block_diff_results)
                 difference_matrix = block_diff_results.get('difference_matrix')
             elif updated_state['previous_roi_stats'] is None:
                 block_change_type = "PRIMEIRA_ANALISE" # Sinaliza que é a primeira vez na ROI
             else:
                 block_change_type = "ROI_INVALIDA/DIF" # Não pôde comparar

        # Atualiza estado para visualização e próximo frame
        updated_state['block_change_type'] = block_change_type
        updated_state['difference_heatmap'] = self.create_difference_heatmap(difference_matrix) if difference_matrix is not None else None
        updated_state['previous_roi_stats'] = current_stats # Guarda stats atuais para próximo frame

        # 3. Máquina de Estados Híbrida
        state = updated_state['state']
        next_state = state # Assume que o estado não muda
        yolo_detected_card = (corners is not None)

        if state == self.WAITING:
            # Inicia se YOLO detecta E análise de bloco indica movimento real (não apenas iluminação)
            # Ou se é a primeira detecção após reset/inicialização
            meaningful_change = block_change_type not in ["NO_CHANGE_OR_LIGHTING", "INICIAL", "PRIMEIRA_ANALISE", "N/A"]
            if yolo_detected_card and meaningful_change:
                print(f"Movimento/Carta detectado (YOLO=True, Bloco={block_change_type}). Estado -> MOTION")
                next_state = self.MOTION
                updated_state['current_card_aligned_box'] = aligned_box
                updated_state['last_stable_corners'] = corners
                updated_state['stability_counter'] = 0
        elif state == self.MOTION:
            if yolo_detected_card:
                # Usa is_same_position (caixa alinhada) para ver se parou
                if self.is_same_position(aligned_box, updated_state['current_card_aligned_box']):
                    print("Carta parece ter parado. Estado -> STABILIZING")
                    next_state = self.STABILIZING
                    updated_state['stability_counter'] = 1
                    # Mantém a caixa alinhada atual, mas atualiza os cantos OBB
                    updated_state['last_stable_corners'] = corners
                else: # Ainda movendo, reseta estabilização
                    # print("Carta ainda em movimento.") # Log opcional
                    updated_state['current_card_aligned_box'] = aligned_box
                    updated_state['last_stable_corners'] = corners
                    updated_state['stability_counter'] = 0
            else: # Perdeu detecção YOLO
                print("Carta (YOLO) desapareceu durante movimento. Estado -> WAITING")
                next_state = self.WAITING
                updated_state['current_card_aligned_box'] = None
                updated_state['last_stable_corners'] = None
                updated_state['stability_counter'] = 0 # Reseta contador
        elif state == self.STABILIZING:
            if yolo_detected_card:
                # Usa is_same_position (caixa alinhada) para ver se continua parado
                if self.is_same_position(aligned_box, updated_state['current_card_aligned_box']):
                    updated_state['stability_counter'] += 1
                    # Atualiza os cantos OBB estáveis a cada frame estável
                    updated_state['last_stable_corners'] = corners
                    
                    # Verifica se precisa esperar cooldown
                    current_time = time.time()
                    last_save_time = updated_state.get('last_save_time', 0)
                    time_since_last_save = current_time - last_save_time
                    cooldown_remaining = max(0, self.save_cooldown - time_since_last_save)
                    cooldown_cleared = time_since_last_save > self.save_cooldown
                    
                    print(f"Estável: {updated_state['stability_counter']}/{self.required_stability_frames}, " +
                         f"Cooldown: {cooldown_cleared}, Restante: {cooldown_remaining:.1f}s")

                    # Atingiu estabilidade E passou cooldown?
                    if updated_state['stability_counter'] >= self.required_stability_frames and cooldown_cleared:
                        print("Carta estável! Recortando e salvando...")

                        # Tenta salvar usando os últimos cantos OBB estáveis
                        saved_path, new_count = self.save_card_with_metadata(
                            frame,
                            updated_state['last_stable_corners'],
                            updated_state['cards_saved_count']
                        )
                        if saved_path:
                            saved_filepath = saved_path # Guarda para retornar
                            updated_state['last_save_time'] = time.time()
                            updated_state['cards_saved_count'] = new_count
                            print(f"Salvo com sucesso! Estado -> WAITING")
                            next_state = self.WAITING
                            # Reseta para próxima detecção
                            updated_state['current_card_aligned_box'] = None
                            updated_state['last_stable_corners'] = None
                            updated_state['stability_counter'] = 0
                        else: # Falha no salvamento
                             print("[ERRO] Falha ao salvar a carta. Estado -> WAITING")
                             next_state = self.WAITING
                             updated_state['current_card_aligned_box'] = None
                             updated_state['last_stable_corners'] = None
                             updated_state['stability_counter'] = 0

                    elif updated_state['stability_counter'] >= self.required_stability_frames:
                         print(f"Estável, mas aguardando cooldown: {cooldown_remaining:.1f}s restantes...")
                         # Mantém STABILIZING, mas não salva ainda

                else: # Moveu de novo durante estabilização
                    print("Carta moveu durante estabilização. Estado -> MOTION")
                    next_state = self.MOTION
                    updated_state['current_card_aligned_box'] = aligned_box
                    updated_state['last_stable_corners'] = corners
                    updated_state['stability_counter'] = 0
            else: # Perdeu detecção YOLO durante estabilização
                print("Carta (YOLO) desapareceu durante estabilização. Estado -> WAITING")
                next_state = self.WAITING
                updated_state['current_card_aligned_box'] = None
                updated_state['last_stable_corners'] = None
                updated_state['stability_counter'] = 0

        # Atualizar estado final
        updated_state['state'] = next_state

        # 4. Visualização
        # Passa o estado atualizado para visualização refletir as últimas mudanças
        processed_frame = self.enhance_visualization(frame, corners, updated_state)

        return processed_frame, updated_state, saved_filepath