import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model("models/unet_multi.tflite")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]  # ou tf.int8 com calibração
tflite_model = converter.convert()

with open("modelo_otimizado.tflite", "wb") as f:
    f.write(tflite_model)
