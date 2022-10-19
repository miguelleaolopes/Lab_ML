import numpy as np

x_train = np.load('data/Xtrain_Classification1.npy')
y_train = np.load('data/Ytrain_Classification1.npy')

# reconstrução de uma das imagens sendo que os dados são listas de valores RBG de uma imagem 30x30

from colorsys import hsv_to_rgb
from PIL import Image

# Make some RGB values. 
# Cycle through hue vertically & saturation horizontally
colors = []
for hue in range(360):
    for sat in range(100):
        # Convert color from HSV to RGB
        rgb = hsv_to_rgb(hue/360, sat/100, 1)
        rgb = [int(0.5 + 255*u) for u in rgb]
        colors.extend(rgb)


# Convert list to bytes ddd
img = Image.frombytes("RGB", (30, 30), x_train[2])
img.show()
img.save('pic1.png')