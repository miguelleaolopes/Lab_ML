import numpy as np

x_train = np.load('data/Xtrain_Classification1.npy')
y_train = np.load('data/Ytrain_Classification1.npy')

# reconstrução de uma das imagens sendo que os dados são listas de valores RBG de uma imagem 30x30

from colorsys import hsv_to_rgb
from PIL import Image


# Convert list to bytes ddd
img = Image.frombytes("RGB", (30, 30), x_train[2])
img.show()
img.save('pic1.png')