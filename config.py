# configuration vars we want to set in one place

# imshape should have 3 channels for rgb input images
# (height, width)
imshape = (576, 640, 3)
# set your classification mode (binary or multi)
mode = 'multi'
# model_name (unet or fcn_8)
model_name = 'unet_'+mode
# log dir for tensorboard
logbase = 'logs'

# classes are defined in hues
# background should be left out
hues = {'pista': 30,
        'carro': 60,
        'gramado': 90,
        'lama': 120,
        'pessoa': 150,
        'cone': 180,
        'obstaculo': 230}

labels = sorted(hues.keys())

if mode == 'binary':
    n_classes = 1

elif mode == 'multi':
    n_classes = len(labels) + 1

assert imshape[0]%32 == 0 and imshape[1]%32 == 0,\
    "imshape should be multiples of 32. comment out to test different imshapes."
