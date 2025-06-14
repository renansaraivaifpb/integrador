from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Conv2D, Input, MaxPooling2D, concatenate, Dropout,\
                                    Lambda, Conv2DTranspose, Add
from config import imshape, n_classes, model_name
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
import numpy as np
import tensorflow as tf
import os


def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x

# Certifique-se de que este decorador está presente
@tf.autograph.experimental.do_not_convert
def dice(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

# --- NOVA MÉTRICA: Intersection over Union (IoU) ---
@tf.autograph.experimental.do_not_convert
def iou(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    # IoU = interseção / (soma_dos_elementos_true + soma_dos_elementos_pred - interseção)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

# --- NOVA MÉTRICA: Pixel Accuracy (Acurácia de Pixel) ---
# O Keras já tem uma métrica de acurácia que funciona para isso,
# mas podemos definir uma explicitamente para controle total ou para referência.
# Para multi-classe (softmax), Keras 'accuracy' já faz isso.
# Para binário (sigmoid), Keras 'binary_accuracy' faz isso.
# No entanto, se quisermos uma função explícita:
@tf.autograph.experimental.do_not_convert
def pixel_accuracy(y_true, y_pred):
    # Para multi-classe (softmax), y_pred precisa ser convertido para labels
    # K.cast converte booleanos (True/False) em float (1.0/0.0)
    # y_true_labels = K.argmax(y_true, axis=-1)
    # y_pred_labels = K.argmax(y_pred, axis=-1)
    # correct_pixels = K.cast(K.equal(y_true_labels, y_pred_labels), K.floatx())
    # return K.mean(correct_pixels)

    # Para ambos os casos (binário/multi-classe) e para Keras 2.x/TF 2.x
    # tf.keras.metrics.Accuracy lida bem com isso.
    # No entanto, se você quiser uma função K.backend, o Keras já tem `categorical_accuracy` ou `binary_accuracy`.
    # Vou usar o Keras 'accuracy' padrão na compilação, pois ele se adapta ao n_classes.
    # Apenas como exemplo de como se poderia fazer:
    # return K.mean(K.equal(K.round(y_true), K.round(y_pred))) # Mais para binário

    # A maneira mais direta de usar a acurácia de pixel para segmentação é usar a métrica 'accuracy' do Keras
    # na compilação do modelo, que já lida com as particularidades de n_classes.
    # Por isso, não é necessário criar uma função customizada 'pixel_accuracy' aqui,
    # basta adicionar 'accuracy' na lista de métricas.
    pass # Este placeholder indica que não estamos criando uma função customizada 'pixel_accuracy' aqui.


def unet(pretrained=False, base=4):

    if pretrained:
        path = os.path.join('models', model_name+'.model')
        if os.path.exists(path):
            # Adicione 'iou' aos custom_objects se você a salvou com o modelo
            model = load_model(path, custom_objects={'dice': dice, 'iou': iou})
            model.summary()
            return model
        else:
            print('Failed to load existing model at: {}'.format(path))

    if n_classes == 1:
        loss = 'binary_crossentropy'
        final_act = 'sigmoid'
        # Para n_classes == 1, a métrica de acurácia de pixel é 'binary_accuracy'
        metrics_list = [dice, iou, 'binary_accuracy']
    elif n_classes > 1:
        loss = 'categorical_crossentropy'
        final_act = 'softmax'
        # Para n_classes > 1, a métrica de acurácia de pixel é 'categorical_accuracy'
        metrics_list = [dice, iou, 'categorical_accuracy']

    b = base
    i = Input((imshape[0], imshape[1], imshape[2]))
    s = Lambda(lambda x: preprocess_input(x)) (i)

    c1 = Conv2D(2**b, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (s)
    c1 = Dropout(0.1) (c1)
    c1 = Conv2D(2**b, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)
    p1 = MaxPooling2D((2, 2)) (c1)

    c2 = Conv2D(2**(b+1), (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
    c2 = Dropout(0.1) (c2)
    c2 = Conv2D(2**(b+1), (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)
    p2 = MaxPooling2D((2, 2)) (c2)

    c3 = Conv2D(2**(b+2), (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)
    c3 = Dropout(0.2) (c3)
    c3 = Conv2D(2**(b+2), (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)
    p3 = MaxPooling2D((2, 2)) (c3)

    c4 = Conv2D(2**(b+3), (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)
    c4 = Dropout(0.2) (c4)
    c4 = Conv2D(2**(b+3), (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

    c5 = Conv2D(2**(b+4), (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p4)
    c5 = Dropout(0.3) (c5)
    c5 = Conv2D(2**(b+4), (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)

    u6 = Conv2DTranspose(2**(b+3), (2, 2), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(2**(b+3), (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)
    c6 = Dropout(0.2) (c6)
    c6 = Conv2D(2**(b+3), (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)

    u7 = Conv2DTranspose(2**(b+2), (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(2**(b+2), (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)
    c7 = Dropout(0.2) (c7)
    c7 = Conv2D(2**(b+2), (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)

    u8 = Conv2DTranspose(2**(b+1), (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(2**(b+1), (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)
    c8 = Dropout(0.1) (c8)
    c8 = Conv2D(2**(b+1), (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)

    u9 = Conv2DTranspose(2**b, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(2**b, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)
    c9 = Dropout(0.1) (c9)
    c9 = Conv2D(2**b, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)

    o = Conv2D(n_classes, (1, 1), activation=final_act) (c9)

    model = Model(inputs=i, outputs=o, name=model_name)
    model.compile(optimizer=Adam(1e-4),
                  loss=loss,
                  metrics=metrics_list) # Usando a lista de métricas
    #model.summary()

    return model


def fcn_8(pretrained=False, base=4):

    if pretrained:
        path = os.path.join('models', model_name+'.model')
        if os.path.exists(path):
            # Adicione 'iou' aos custom_objects se você a salvou com o modelo
            model = load_model(path, custom_objects={'dice': dice, 'iou': iou})
            return model
        else:
            print('Failed to load existing model at: {}'.format(path))

    if n_classes == 1:
        loss = 'binary_crossentropy'
        final_act = 'sigmoid'
        metrics_list = [dice, iou, 'binary_accuracy']
    elif n_classes > 1:
        loss = 'categorical_crossentropy'
        final_act = 'softmax'
        metrics_list = [dice, iou, 'categorical_accuracy']

    b = base
    i = Input(shape=imshape)
    s = Lambda(lambda x: preprocess_input(x)) (i)
    ## Block 1
    x = Conv2D(2**b, (3, 3), activation='elu', padding='same', name='block1_conv1')(s)
    x = Conv2D(2**b, (3, 3), activation='elu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    f1 = x

    # Block 2
    x = Conv2D(2**(b+1), (3, 3), activation='elu', padding='same', name='block2_conv1')(x)
    x = Conv2D(2**(b+1), (3, 3), activation='elu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    f2 = x

    # Block 3
    x = Conv2D(2**(b+2), (3, 3), activation='elu', padding='same', name='block3_conv1')(x)
    x = Conv2D(2**(b+2), (3, 3), activation='elu', padding='same', name='block3_conv2')(x)
    x = Conv2D(2**(b+2), (3, 3), activation='elu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    pool3 = x

    # Block 4
    x = Conv2D(2**(b+3), (3, 3), activation='elu', padding='same', name='block4_conv1')(x)
    x = Conv2D(2**(b+3), (3, 3), activation='elu', padding='same', name='block4_conv2')(x)
    x = Conv2D(2**(b+3), (3, 3), activation='elu', padding='same', name='block4_conv3')(x)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(2**(b+3), (3, 3), activation='elu', padding='same', name='block5_conv1')(pool4)
    x = Conv2D(2**(b+3), (3, 3), activation='elu', padding='same', name='block5_conv2')(x)
    x = Conv2D(2**(b+3), (3, 3), activation='elu', padding='same', name='block5_conv3')(x)
    pool5 = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    conv6 = Conv2D(2048 , (7, 7) , activation='elu' , padding='same', name="conv6")(pool5)
    conv6 = Dropout(0.5)(conv6)
    conv7 = Conv2D(2048 , (1, 1) , activation='elu' , padding='same', name="conv7")(conv6)
    conv7 = Dropout(0.5)(conv7)

    pool4_n = Conv2D(n_classes, (1, 1), activation='elu', padding='same')(pool4)
    u2 = Conv2DTranspose(n_classes, kernel_size=(2, 2), strides=(2, 2), padding='same')(conv7)
    u2_skip = Add()([pool4_n, u2])

    pool3_n = Conv2D(n_classes, (1, 1), activation='elu', padding='same')(pool3)
    u4 = Conv2DTranspose(n_classes, kernel_size=(2, 2), strides=(2, 2), padding='same')(u2_skip)
    u4_skip = Add()([pool3_n, u4])

    o = Conv2DTranspose(n_classes, kernel_size=(8, 8), strides=(8, 8), padding='same',
                        activation=final_act)(u4_skip)

    model = Model(inputs=i, outputs=o, name=model_name)
    model.compile(optimizer=Adam(1e-4),
                  loss=loss,
                  metrics=metrics_list) # Usando a lista de métricas
    #model.summary()

    return model
