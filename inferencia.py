import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from utils import add_masks, crf
from config import imshape, n_classes

# === CONFIGURAÇÕES ===
MODE = 'softmax'          # ou 'argmax'
CALC_CRF = False          # Ativar CRF?
BACKGROUND = True         # Fundir máscara com imagem original?
INPUT_FOLDER = 'logs/images_to_predict'
OUTPUT_FOLDER = 'outputs_inferencias'
NUM_IMAGES = 60            # Quantidade de imagens para inferência

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --- CARREGA O MODELO TFLITE ---
tflite_model_path = 'models/unet_multi.tflite'
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_index = input_details[0]['index']
input_shape = input_details[0]['shape']

# Filtra imagens válidas
valid_images = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
if not valid_images:
    print("Nenhuma imagem válida encontrada.")
    exit()

# Limita à quantidade desejada
valid_images = valid_images[:NUM_IMAGES]

# Loop de inferência
for i, filename in enumerate(valid_images, 1):
    image_path = os.path.join(INPUT_FOLDER, filename)

    # Carrega e prepara a imagem
    image = cv2.imread(image_path)
    image_resized = cv2.resize(image, (imshape[1], imshape[0]))
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)

    # Pré-processamento (ajuste conforme necessário)
    input_tensor = np.expand_dims(image_rgb.astype(np.float32), axis=0)
    # Exemplo de normalização (descomente se necessário)
    # input_tensor = input_tensor / 255.0

    if input_tensor.shape != tuple(input_shape):
        input_tensor = np.resize(input_tensor, input_shape).astype(input_details[0]['dtype'])
    else:
        input_tensor = input_tensor.astype(input_details[0]['dtype'])

    interpreter.set_tensor(input_index, input_tensor)
    interpreter.invoke()

    output_index = output_details[0]['index']
    pred = interpreter.get_tensor(output_index)

    print(f"[{i}/{NUM_IMAGES}] Shape da predição:", pred.shape)

    # Pós-processamento
    if MODE == 'argmax':
        if n_classes == 1:
            pred = pred.squeeze()
            softmax = np.stack([1 - pred, pred], axis=2)
            pred = to_categorical(np.argmax(softmax, axis=2))
        else:
            pred = np.argmax(pred.squeeze(), axis=2)
            pred = to_categorical(pred, num_classes=n_classes)

    if CALC_CRF:
        if n_classes == 1:
            softmax = np.stack([1 - pred.squeeze(), pred.squeeze()], axis=2)
            mask = crf(softmax, image_rgb)
            mask = cv2.cvtColor(np.array(mask, dtype=np.float32), cv2.COLOR_GRAY2RGB)
        else:
            mask = crf(pred.squeeze(), image_rgb)
    else:
        if n_classes == 1:
            mask = pred.squeeze() * 255.0
            mask = cv2.cvtColor(mask.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        else:
            mask = add_masks(pred.squeeze() * 255.0)

    # Salva a máscara separadamente
    mask_resized = cv2.resize(mask.astype(np.uint8), (image_resized.shape[1], image_resized.shape[0]))
    mask_save_path = os.path.join(OUTPUT_FOLDER, f'mask_{filename}')
    cv2.imwrite(mask_save_path, cv2.cvtColor(mask_resized, cv2.COLOR_RGB2BGR))
    print(f"   Máscara salva em: {mask_save_path}")

    # Combina imagem + máscara (se ativado)
    if BACKGROUND:
        blended = cv2.addWeighted(image_rgb, 1.0, mask_resized, 1.0, 0)
        result = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)
    else:
        result = cv2.cvtColor(mask_resized, cv2.COLOR_RGB2BGR)

    result_path = os.path.join(OUTPUT_FOLDER, f'result_{filename}')
    cv2.imwrite(result_path, result)
    print(f"   Resultado salvo em: {result_path}")
