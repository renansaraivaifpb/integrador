import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from utils import add_masks, crf
from models import dice
from config import imshape, model_name, n_classes

# === CONFIGURAÇÕES ===
MODE = 'softmax'          # ou 'argmax'
CALC_CRF = False          # Ativar CRF?
BACKGROUND = True         # Fundir máscara com imagem original?
INPUT_FOLDER = 'C:/Users/arqis/Documents/renan/integrador-main/logs/images_to_predict'
OUTPUT_FOLDER = 'outputs_inferencias'

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Carrega o modelo
model = load_model('models/unetmulti', custom_objects={'dice': dice}, compile=False)

# Encontra a primeira imagem válida na pasta
image_path = None
for filename in os.listdir(INPUT_FOLDER):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(INPUT_FOLDER, filename)
        break

if image_path is None:
    print("Nenhuma imagem encontrada na pasta.")
    exit()

# Carrega e prepara a imagem
image = cv2.imread(image_path)
image_resized = cv2.resize(image, (imshape[1], imshape[0]))
image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
input_tensor = np.expand_dims(image_rgb, axis=0)

# Predição
pred = model.predict(input_tensor)

# Diagnóstico
print("Shape da predição:", pred.shape)
print("Valor mínimo:", np.min(pred))
print("Valor máximo:", np.max(pred))

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
mask_save_path = os.path.join(OUTPUT_FOLDER, 'mask_' + os.path.basename(image_path))
cv2.imwrite(mask_save_path, cv2.cvtColor(mask_resized, cv2.COLOR_RGB2BGR))
print(f"Máscara salva separadamente em: {mask_save_path}")

# Combina imagem + máscara (se ativado)
if BACKGROUND:
    blended = cv2.addWeighted(image_rgb, 1.0, mask_resized, 1.0, 0)
    result = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)
else:
    result = cv2.cvtColor(mask_resized, cv2.COLOR_RGB2BGR)

# Salva imagem final
save_path = os.path.join(OUTPUT_FOLDER, 'result_' + os.path.basename(image_path))
cv2.imwrite(save_path, result)
print(f"Inferência feita em: {image_path}")
print(f"Resultado salvo em: {save_path}")
