import numpy as np
from colorsys import hsv_to_rgb
from PIL import Image


x_import = np.load('data/Xtrain_Classification1.npy')
y_import = np.load('data/Ytrain_Classification1.npy')


# Convert list to bytes ddd
for i in range(8273):
    img = Image.frombytes("RGB", (30, 30), x_import[i])
    if y_import[i] == 1: #eyespot
        img.save('images/eyespot/pic{}.png'.format(i))
    else:
        img.save('images/spot/pic{}.png'.format(i))

print('done')