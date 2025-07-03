import tensorflow as tf
from tensorflow.keras.models import load_model
from models import dice

model = load_model('models/unet_multi_tf2_compat.h5', custom_objects={'dice': dice}, compile=False)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('unet_multi.tflite', 'wb') as f:
    f.write(tflite_model)
