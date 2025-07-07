from tensorflow.keras.models import load_model
from models import dice

model = load_model('models/unet_multi.h5', custom_objects={'dice': dice}, compile=False)
model.save('models/unet_multi_tf2_compat.h5')  # reexporta
