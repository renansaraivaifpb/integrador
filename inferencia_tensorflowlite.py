import os
import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter
from config import imshape, n_classes, labels, hues

# === CONFIGURAÇÕES ===
MODE = 'softmax'          # ou 'argmax'
CALC_CRF = False          # Ativar CRF?
BACKGROUND = True         # Fundir máscara com imagem original?
INPUT_FOLDER = 'logs/images_to_predict'
OUTPUT_FOLDER = 'outputs_inferencias'
NUM_IMAGES = 60

os.makedirs(OUTPUT_FOLDER, exist_ok=True)




def add_masks(pred):
    blank = np.zeros(shape=imshape, dtype=np.uint8)

    for i, label in enumerate(labels):

        hue = np.full(shape=(imshape[0], imshape[1]), fill_value=hues[label], dtype=np.uint8)
        sat = np.full(shape=(imshape[0], imshape[1]), fill_value=255, dtype=np.uint8)
        val = pred[:,:,i].astype(np.uint8)

        im_hsv = cv2.merge([hue, sat, val])
        im_rgb = cv2.cvtColor(im_hsv, cv2.COLOR_HSV2RGB)
        blank = cv2.add(blank, im_rgb)

    return blank


def crf(im_softmax, im_rgb):
    n_classes = im_softmax.shape[2]
    feat_first = im_softmax.transpose((2, 0, 1)).reshape(n_classes, -1)
    unary = unary_from_softmax(feat_first)
    unary = np.ascontiguousarray(unary)
    im_rgb = np.ascontiguousarray(im_rgb)

    d = dcrf.DenseCRF2D(im_rgb.shape[1], im_rgb.shape[0], n_classes)

    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=(5, 5), compat=3, kernel=dcrf.DIAG_KERNEL,
                              normalization=dcrf.NORMALIZE_SYMMETRIC)

    # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
    d.addPairwiseBilateral(sxy=(5, 5), srgb=(13, 13, 13), rgbim=im_rgb,
                           compat=10,
                           kernel=dcrf.DIAG_KERNEL,
                           normalization=dcrf.NORMALIZE_SYMMETRIC)
    Q = d.inference(5)
    res = np.argmax(Q, axis=0).reshape((im_rgb.shape[0], im_rgb.shape[1]))
    if mode == 'binary':
        return res * 255.0
    if mode =='multi':
        res_hot = to_categorical(res) * 255.0
        res_crf = add_masks(res_hot)
        return res_crf

# --- Função alternativa para to_categorical ---
def to_categorical(y, num_classes=None):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if num_classes is None:
        num_classes = np.max(y) + 1
    y_flat = y.ravel()
    categorical = np.zeros((y_flat.shape[0], num_classes), dtype='float32')
    categorical[np.arange(y_flat.shape[0]), y_flat] = 1
    categorical = categorical.reshape(*input_shape, num_classes)
    return categorical

# --- CARREGA O MODELO TFLITE ---
tflite_model_path = 'models/unet_multi.tflite'
interpreter = Interpreter(model_path=tflite_model_path)
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

valid_images = valid_images[:NUM_IMAGES]

# Loop de inferência
for i, filename in enumerate(valid_images, 1):
    image_path = os.path.join(INPUT_FOLDER, filename)

    image = cv2.imread(image_path)
    image_resized = cv2.resize(image, (imshape[1], imshape[0]))
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)

    input_tensor = np.expand_dims(image_rgb.astype(np.float32), axis=0)

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

    mask_resized = cv2.resize(mask.astype(np.uint8), (image_resized.shape[1], image_resized.shape[0]))
    mask_save_path = os.path.join(OUTPUT_FOLDER, f'mask_{filename}')
    cv2.imwrite(mask_save_path, cv2.cvtColor(mask_resized, cv2.COLOR_RGB2BGR))
    print(f"   Máscara salva em: {mask_save_path}")

    if BACKGROUND:
        blended = cv2.addWeighted(image_rgb, 1.0, mask_resized, 1.0, 0)
        result = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)
    else:
        result = cv2.cvtColor(mask_resized, cv2.COLOR_RGB2BGR)

    result_path = os.path.join(OUTPUT_FOLDER, f'result_{filename}')
    cv2.imwrite(result_path, result)
    print(f"   Resultado salvo em: {result_path}")
