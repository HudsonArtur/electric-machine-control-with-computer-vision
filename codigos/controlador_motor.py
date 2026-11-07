import cv2
import time
import mediapipe as mp
import serial
import math
import numpy as np

# Importa a interface e as paletas do seu arquivo ar_hand_2.py
from ar_hand import draw_ar_ui, draw_hand_skeleton, COLOR_PALETTES

# Limites do ângulo (em graus) para mapear a velocidade
ANGULO_MIN = 0
ANGULO_MAX = 90
ultimo_valor_enviado = -1

# Configura a Conexão com o ESP32
porta_esp32 = 'COM8'
try:
    conexao = serial.Serial(
        port=porta_esp32,
        baudrate=115200,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        timeout=1,
        write_timeout=1, # <-- ADICIONE ESTA LINHA
        dsrdtr=False,
        rtscts=False
    )
    print(f"Conectado ao ESP32 na porta {porta_esp32}")
except Exception as e:
    print(f"Erro na porta serial: {e}")
    exit()

# Configurações do MediaPipe Tasks
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

def draw_dimmer_display(frame, velocidade, width):
    padding = 20
    text_x = width - 620 - padding 
    text_y_pwm =  padding
    texto_pwm = f"Velocidade do motor: {(velocidade/255*100):.2f}%"
    cor_brilho = (0, int(velocidade), int(255 - velocidade))
    cv2.putText(frame, texto_pwm, (text_x, text_y_pwm), cv2.FONT_HERSHEY_SIMPLEX, 0.9, cor_brilho, 2, cv2.LINE_AA)

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=r"C:/Users/Hudson Artur/Documents/Faculdade/TCC_Engenharia_Eletrica/codigos_midiapipe/hand_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1
)

# Estado da aplicação para controlar o toggle de cores da interface RA
app_state = {
    'prev_fist': False,
    'color_index': 0,
    'cooldown': 0
}

with HandLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)
    start_time = time.time()
    
    if cap.isOpened():
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    else:
        width, height = 640, 480

    ultimo_tempo_envio = 0 

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1) 
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        timestamp_ms = int((time.time() - start_time) * 1000)

        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.hand_landmarks:
            landmarks_da_mao = result.hand_landmarks[0]
            
            # Aplica a interface AR
            draw_ar_ui(frame, landmarks_da_mao, app_state)
            draw_hand_skeleton(frame, landmarks_da_mao)
            
            # ==================================================================
            # CÁLCULO DA ANGULAÇÃO DA MÃO (Substituindo a pinça)
            # ==================================================================
            p0 = landmarks_da_mao[0] # Pulso
            p9 = landmarks_da_mao[9] # Base do dedo médio
            
            p0_x, p0_y = p0.x * width, p0.y * height
            p9_x, p9_y = p9.x * width, p9.y * height
            
            v_palm_x = p9_x - p0_x
            v_palm_y = p9_y - p0_y
            
            angle_rad = np.abs(math.atan2(-v_palm_x, -v_palm_y))
            angle_deg = int(math.degrees(angle_rad))
            
            # ==================================================================
            # CÁLCULO DA VELOCIDADE BASEADO NO ÂNGULO
            # ==================================================================
            # Limita o ângulo capturado aos limites definidos
            angulo_limitado = max(ANGULO_MIN, min(ANGULO_MAX, angle_deg))
            
            # Mapeia o ângulo limitado (ex: 0 a 90) para o PWM (0 a 255)
            velocidade = int(((angulo_limitado - ANGULO_MIN) / (ANGULO_MAX - ANGULO_MIN)) * 255)

            # ==================================================================
            # ENVIO SERIAL
            # ==================================================================
            tempo_atual = time.time()
            
            if (tempo_atual - ultimo_tempo_envio) >= 0.05:
                if abs(velocidade - ultimo_valor_enviado) >= 3:
                    try:
                        conexao.write(f"{velocidade}\n".encode())
                        ultimo_valor_enviado = velocidade
                        ultimo_tempo_envio = tempo_atual
                        print(f"-> Angulo: {angle_deg}º | Vel: {velocidade}")
                    except serial.SerialTimeoutException:
                        print("Aviso: Timeout ao enviar dados para o ESP32 (Possível ruído ou desconexão).")
                    except Exception as e:
                        print(f"Erro serial inesperado: {e}")
            
            draw_dimmer_display(frame, velocidade, width)

        cv2.imshow('Controle de Motor por Rotacao', frame)
        
        if cv2.waitKey(5) & 0xFF == 27: 
            conexao.write(b"0\n") 
            print("\nCâmera interrompida com segurança.")
            break

cap.release()
cv2.destroyAllWindows()
conexao.close()