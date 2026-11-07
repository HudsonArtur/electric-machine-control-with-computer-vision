import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import os
import urllib.request
import math

# ==============================================================================
# CONFIGURAÇÕES E CONSTANTES
# ==============================================================================

# Conexões anatômicas do MediaPipe (índices dos landmarks que se ligam)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),         # Polegar
    (0, 5), (5, 6), (6, 7), (7, 8),         # Indicador
    (5, 9), (9, 10), (10, 11), (11, 12),    # Médio
    (9, 13), (13, 14), (14, 15), (15, 16),  # Anelar
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) # Mínimo e base da palma
]

# Cores para a interface (formato BGR)
CYAN = (255, 255, 0)
ORANGE = (0, 180, 255)
WHITE = (255, 255, 255)
RED = (0, 0, 255)
DARK_RED = (0, 0, 150)
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
PURPLE = (255, 0, 255)
PINK = (255, 0, 128)

# Paletas de cores que serão alternadas
COLOR_PALETTES = [
    (CYAN, ORANGE),      # Paleta 0: Ciano e Laranja
    (RED, DARK_RED),     # Paleta 1: Vermelho e Vermelho Escuro
    (GREEN, YELLOW),     # Paleta 2: Verde e Amarelo
    (PURPLE, PINK)       # Paleta 3: Roxo e Rosa
]

# Caminho para o modelo do MediaPipe
MODEL_PATH = 'hand_landmarker.task'
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

# ==============================================================================
# FUNÇÕES DE DESENHO DA INTERFACE AR
# ==============================================================================

def draw_hand_skeleton(img, hand_landmarks):
    """Desenha os ossos e as articulações da mão."""
    h, w = img.shape[:2]
    
    pixel_landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
    
    for connection in HAND_CONNECTIONS:
        start_point = pixel_landmarks[connection[0]]
        end_point = pixel_landmarks[connection[1]]
        cv2.line(img, start_point, end_point, WHITE, 1, cv2.LINE_AA)
        
    for idx, point in enumerate(pixel_landmarks):
        if idx in [4, 8, 12, 16, 20]:
            cv2.circle(img, point, 5, ORANGE, -1, cv2.LINE_AA)
        else:
            cv2.circle(img, point, 4, RED, -1, cv2.LINE_AA)
            cv2.circle(img, point, 4, ORANGE, 1, cv2.LINE_AA)

def draw_glow_circle(img, center, radius, color, thickness=2, glow=15):
    """Desenha um círculo com efeito de brilho."""
    for g in range(glow, 0, -3):
        alpha = 0.08 + 0.12 * (g / glow)
        overlay = img.copy()
        cv2.circle(overlay, center, radius + g, color, thickness)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.circle(img, center, radius, color, thickness)

def draw_radial_ticks(img, center, radius, color, num_ticks=24, length=22, thickness=3):
    """Desenha marcações radiais em torno de um círculo."""
    for i in range(num_ticks):
        angle = np.deg2rad(i * (360 / num_ticks))
        x1 = int(center[0] + (radius - length) * np.cos(angle))
        y1 = int(center[1] + (radius - length) * np.sin(angle))
        x2 = int(center[0] + radius * np.cos(angle))
        y2 = int(center[1] + radius * np.sin(angle))
        cv2.line(img, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

def draw_core_pattern(img, center, radius, main_color, accent_color):
    """Desenha um padrão estilizado no centro do HUD."""
    for t in np.linspace(0, 2 * np.pi, 40):
        r = radius * (0.7 + 0.3 * np.sin(6 * t))
        x = int(center[0] + r * np.cos(t))
        y = int(center[1] + r * np.sin(t))
        cv2.circle(img, (x, y), 3, accent_color, -1, cv2.LINE_AA)
    
    cv2.circle(img, center, int(radius * 0.6), main_color, 2, cv2.LINE_AA)
    cv2.circle(img, center, int(radius * 0.4), accent_color, 2, cv2.LINE_AA)

def draw_hud_details(img, center, main_color):
    """Desenha detalhes adicionais do HUD (barras e blocos)."""
    for i in range(8):
        angle = np.deg2rad(210 + i * 10)
        x1 = int(center[0] + 140 * np.cos(angle))
        y1 = int(center[1] + 140 * np.sin(angle))
        x2 = int(center[0] + 170 * np.cos(angle))
        y2 = int(center[1] + 170 * np.sin(angle))
        cv2.line(img, (x1, y1), (x2, y2), main_color, 4, cv2.LINE_AA)
    
    for i in range(4):
        angle = np.deg2rad(270 + i * 15)
        x = int(center[0] + 120 * np.cos(angle))
        y = int(center[1] + 120 * np.sin(angle))
        cv2.rectangle(img, (x - 10, y - 10), (x + 10, y + 10), main_color, 2)

def draw_arc_segments(img, center, main_color, accent_color):
    """Desenha segmentos de arco no HUD."""
    cv2.ellipse(img, center, (110, 110), 0, -30, 210, main_color, 3, cv2.LINE_AA)
    cv2.ellipse(img, center, (100, 100), 0, -30, 210, accent_color, 2, cv2.LINE_AA)

def draw_ar_ui(frame, hand_landmarks, app_state):
    """Desenha toda a interface de RA com base nos pontos da mão."""
    h, w = frame.shape[:2]
    
    # Ponto central (base do dedo médio - centro da palma)
    middle_mcp = hand_landmarks[9]
    # Ponto de referência (pulso) para calcular o tamanho
    wrist = hand_landmarks[0]
    
    # Média dos nós da palma (pulso, base do indicador, médio, anelar e mínimo)
    palm_indices = [0, 5, 9, 13, 17]
    center_x = int(sum(hand_landmarks[i].x for i in palm_indices) / len(palm_indices) * w)
    center_y = int(sum(hand_landmarks[i].y for i in palm_indices) / len(palm_indices) * h)
    center = (center_x, center_y)
    ref_point = (int(wrist.x * w), int(wrist.y * h))
    
    # Calcula o raio com base na distância entre o centro da palma e o pulso
    distance = np.linalg.norm(np.array(center) - np.array(ref_point))
    radius = int(distance * 1.5)  
    
    # ==========================================================================
    # DETECÇÃO DE MÃO FECHADA (PUNHO) E LÓGICA DE TOGGLE
    # ==========================================================================
    wrist_px = np.array([wrist.x * w, wrist.y * h])
    
    def dist_to_wrist(idx):
        pt = np.array([hand_landmarks[idx].x * w, hand_landmarks[idx].y * h])
        return np.linalg.norm(pt - wrist_px)
    
    index_closed = dist_to_wrist(8) < dist_to_wrist(5)
    middle_closed = dist_to_wrist(12) < dist_to_wrist(9)
    ring_closed = dist_to_wrist(16) < dist_to_wrist(13)
    pinky_closed = dist_to_wrist(20) < dist_to_wrist(17)
    
    is_fist = index_closed and middle_closed and ring_closed and pinky_closed
    
    # Lógica de Toggle: Muda a cor apenas no momento em que a mão FECHA
    if app_state['cooldown'] > 0:
        app_state['cooldown'] -= 1
    else:
        # Se a mão está fechada AGORA, mas estava ABERTA no frame anterior
        if is_fist and not app_state['prev_fist']:
            # Avança para a próxima paleta de cores (volta ao início se chegar ao fim)
            app_state['color_index'] = (app_state['color_index'] + 1) % len(COLOR_PALETTES)
            # Ativa um cooldown de 15 frames para evitar toggles acidentais por ruído
            app_state['cooldown'] = 15
        
        # Atualiza o estado anterior para o próximo frame
        app_state['prev_fist'] = is_fist
    
    # Seleciona as cores da paleta atual
    main_color, accent_color = COLOR_PALETTES[app_state['color_index']]
    
    # Desenha os elementos da interface com as cores dinâmicas
    draw_glow_circle(frame, center, radius, main_color, 3)
    draw_radial_ticks(frame, center, radius, main_color)
    draw_core_pattern(frame, center, radius, main_color, accent_color)
    draw_hud_details(frame, center, main_color)
    draw_arc_segments(frame, center, main_color, accent_color)
    
    # ==========================================================================
    # CÁLCULO DO ÂNGULO DE ROTAÇÃO DA MÃO (ESQUERDA/DIREITA)
    # ==========================================================================
    p0_px = np.array([hand_landmarks[0].x * w, hand_landmarks[0].y * h])
    p9_px = np.array([hand_landmarks[9].x * w, hand_landmarks[9].y * h])
    
    v_palm = p9_px - p0_px
    
    angle_rad = math.atan2(-v_palm[0], -v_palm[1])
    angle_deg = math.degrees(angle_rad)
    
    # Desenha o texto do ângulo perto do pulso
    text_position = (int(p0_px[0] + 20), int(p0_px[1] - 20))
    cv2.putText(frame, f"Rotacao: {int(angle_deg)} graus", text_position,
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, main_color, 2, cv2.LINE_AA)
    
    # Desenha uma linha de referência vertical (apontando para cima a partir do pulso)
    ref_length = 80
    p_ref = (int(p0_px[0]), int(p0_px[1] - ref_length))
    cv2.line(frame, tuple(p0_px.astype(int)), p_ref, WHITE, 1, cv2.LINE_AA)
    
    # Desenha a linha da palma (do pulso até a base do dedo médio)
    cv2.line(frame, tuple(p0_px.astype(int)), tuple(p9_px.astype(int)), main_color, 2, cv2.LINE_AA)
    
    # Desenha o arco do ângulo no pulso
    start_angle = -90 
    end_angle = start_angle + angle_deg
    cv2.ellipse(frame, tuple(p0_px.astype(int)), (60, 60), 0, start_angle, end_angle, main_color, 2, cv2.LINE_AA)

    # Exibe o modo de cor atual no canto superior esquerdo (abaixo do FPS)
    cv2.putText(frame, f"Modo de Operacao: {app_state['color_index'] + 1}/{len(COLOR_PALETTES)}", 
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, main_color, 2, cv2.LINE_AA)

# ==============================================================================
# FUNÇÕES AUXILIARES
# ==============================================================================

def download_model():
    """Baixa o modelo hand_landmarker.task se não existir."""
    if not os.path.exists(MODEL_PATH):
        print("Baixando o modelo hand_landmarker.task (~12 MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Download concluído.")

def landmark_px(landmark, w, h):
    """Converte coordenadas normalizadas do landmark para pixels."""
    return int(landmark.x * w), int(landmark.y * h)

# ==============================================================================
# LOOP PRINCIPAL
# ==============================================================================

def main():
    download_model()
    
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7
    )
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erro: Não foi possível abrir a câmera.")
        return
    
    prev_tick = cv2.getTickCount()
    start_time = time.time()
    
    # Estado da aplicação para controlar o toggle de cores
    app_state = {
        'prev_fist': False,  # Estado anterior da mão (fechada/aberta)
        'color_index': 0,    # Índice da paleta de cores atual
        'cooldown': 0        # Contador para evitar toggles rápidos
    }
    
    print("Pressione 'q' para sair.")
    
    with vision.HandLandmarker.create_from_options(options) as detector:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Ignorando frame vazio da câmera.")
                continue
            
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            
            timestamp_ms = int((time.time() - start_time) * 1000)
            detection_result = detector.detect_for_video(mp_image, timestamp_ms)
            
            if detection_result.hand_landmarks:
                for hand_landmarks in detection_result.hand_landmarks:
                    draw_ar_ui(frame, hand_landmarks, app_state)
                    draw_hand_skeleton(frame, hand_landmarks)
            
            curr_tick = cv2.getTickCount()
            fps = cv2.getTickFrequency() / (curr_tick - prev_tick)
            prev_tick = curr_tick
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
            
            cv2.imshow("Hand Tracking AR UI", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()