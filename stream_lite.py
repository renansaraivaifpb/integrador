import os
import time
import contextlib
import numpy as np
import cv2
import pygame
from utils import VideoStream
from models import preprocess_input
from config import imshape, model_name, n_classes
from utils import add_masks, crf
import tflite_runtime.interpreter as tflite

# Variáveis de controle da execução e do modo de funcionamento
RUN = True               # Controle do loop principal
MODE = 'softmax'         # Modo de saída do modelo: softmax ou argmax
CALC_CRF = False         # Flag para ativar/desativar pós-processamento CRF
BACKGROUND = False       # Flag para ativar/desativar sobreposição do background

# Definição das dimensões do frame e região de interesse (ROI)
frame_shape = (640, 480)                 # Resolução da janela do Pygame
target_shape = imshape[:2]               # Resolução esperada pelo modelo (altura, largura)
d_width = target_shape[0] // 2           # Metade da largura do ROI
d_height = target_shape[1] // 2          # Metade da altura do ROI
x0 = frame_shape[1] // 2 - d_width       # Coordenada X inicial do ROI (centralizado)
y0 = frame_shape[0] // 2 - d_height      # Coordenada Y inicial do ROI (centralizado)
x1 = frame_shape[1] // 2 + d_width       # Coordenada X final do ROI
y1 = frame_shape[0] // 2 + d_height      # Coordenada Y final do ROI

# Inicializa o interpretador do TensorFlow Lite com o modelo carregado
interpreter = tflite.Interpreter(model_path=os.path.join('models', model_name + '.tflite'))
interpreter.allocate_tensors()               # Aloca tensores para entrada e saída
input_details = interpreter.get_input_details()   # Detalhes da entrada do modelo
output_details = interpreter.get_output_details() # Detalhes da saída do modelo

# Inicialização do Pygame para exibição da interface
pygame.init()
screen = pygame.display.set_mode(frame_shape)   # Cria janela com a resolução definida

# Inicializa a captura de vídeo da webcam
vs = VideoStream(device=0).start()
time.sleep(0.1)     # Pequena pausa para garantir que a câmera iniciou
prev = time.time()  # Marca o tempo inicial para cálculo de FPS

# Loop principal de processamento de frames
while RUN:

    current = time.time()    # Tempo atual para cálculo de delta (tempo entre frames)

    if vs.check_queue():     # Verifica se há um frame disponível na fila

        delta = current - prev
        prev = current

        screen.fill([0, 0, 0])                   # Limpa tela Pygame com preto
        frame = vs.read()                        # Captura frame da webcam
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Converte BGR (OpenCV) para RGB (Pygame)
        im = frame.copy()                        # Cópia do frame para processamento

        # Recorta a região de interesse (ROI) central do frame
        roi = im[x0:x1, y0:y1]
        tmp = np.expand_dims(roi, axis=0).astype(np.float32)  # Adiciona dimensão batch e converte para float32

        # Pré-processa a entrada conforme requerido pelo modelo (normalização, etc)
        tmp = preprocess_input(tmp)

        # Realiza a inferência com o modelo TensorFlow Lite
        interpreter.set_tensor(input_details[0]['index'], tmp)  # Define input do modelo
        interpreter.invoke()                                     # Executa inferência
        roi_pred = interpreter.get_tensor(output_details[0]['index'])  # Obtém output do modelo

        # Processa o output do modelo conforme modo selecionado
        if MODE == 'argmax':
            if n_classes == 1:
                roi_pred = roi_pred.squeeze()
                roi_softmax = np.stack([1 - roi_pred, roi_pred], axis=2)  # Cria softmax manual para 2 classes
                roi_max = np.argmax(roi_softmax, axis=2)
                roi_pred = np.eye(2)[roi_max]                            # One-hot encoding da predição
            elif n_classes > 1:
                roi_max = np.argmax(roi_pred.squeeze(), axis=2)
                roi_pred = np.eye(n_classes)[roi_max]

        # Pós-processamento com CRF para melhorar máscara, se ativado
        if CALC_CRF:
            if n_classes == 1:
                roi_pred = roi_pred.squeeze()
                roi_softmax = np.stack([1 - roi_pred, roi_pred], axis=2)
                roi_mask = crf(roi_softmax, roi)                         # Aplica CRF
                roi_mask = np.array(roi_mask, dtype=np.float32)
                roi_mask = cv2.cvtColor(roi_mask, cv2.COLOR_GRAY2RGB)   # Converte para 3 canais
            elif n_classes > 1:
                roi_mask = crf(roi_pred.squeeze(), roi)
        else:
            # Sem CRF, cria máscara simples a partir da predição
            if n_classes == 1:
                roi_mask = roi_pred.squeeze() * 255.0
                roi_mask = cv2.cvtColor(roi_mask, cv2.COLOR_GRAY2RGB)
            elif n_classes > 1:
                roi_mask = add_masks(roi_pred.squeeze() * 255.0)         # Sobrepõe máscaras coloridas

        # Se background ativado, sobrepõe máscara colorida sobre ROI original
        if BACKGROUND:
            roi_mask = np.array(roi_mask, dtype=np.uint8)
            roi_mask = cv2.addWeighted(roi, 1.0, roi_mask, 1.0, 0)

        # Atualiza o frame original com a máscara gerada na ROI
        frame[x0:x1, y0:y1] = roi_mask

        # Desenha um retângulo vermelho ao redor da ROI
        cv2.rectangle(frame, (y0, x0), (y1, x1), (0, 0, 255), 2)

        # Adiciona textos informativos na tela: FPS, modo e CRF
        cv2.putText(frame, 'FPS: ' + str(np.round(1 / delta, 1)), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, 'MODE: ' + str(MODE), (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, 'CRF: ' + str(CALC_CRF), (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)

        # Ajusta orientação para exibição no Pygame (flip e rotação)
        frame = cv2.flip(frame, 0)
        frame = np.rot90(frame, k=3)

        # Converte o frame para surface do Pygame e desenha na tela
        frame = pygame.surfarray.make_surface(frame)
        screen.blit(frame, (0, 0))
        pygame.display.update()

        # Tratamento de eventos do teclado e janela
        for event in pygame.event.get():

            keys = pygame.key.get_pressed()

            if event.type == pygame.QUIT:    # Fecha janela se fechar
                RUN = False
            elif event.type == pygame.KEYDOWN and keys[pygame.K_q]:  # Tecla Q encerra o programa
                RUN = False
            elif event.type == pygame.KEYDOWN and keys[pygame.K_c]:  # Tecla C ativa/desativa CRF
                CALC_CRF = not CALC_CRF
            elif event.type == pygame.KEYDOWN and keys[pygame.K_m]:  # Tecla M alterna modo softmax/argmax
                MODE = 'argmax' if MODE == 'softmax' else 'softmax'
            elif event.type == pygame.KEYDOWN and keys[pygame.K_b]:  # Tecla B ativa/desativa sobreposição background
                BACKGROUND = not BACKGROUND

# Finaliza tudo ao sair do loop
cv2.destroyAllWindows()
vs.stop()
