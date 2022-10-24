import numpy as np
from matplotlib import pyplot as plt

x_train = np.load('data/Xtrain_Classification1.npy')
y_train = np.load('data/Ytrain_Classification1.npy')



# reconstrução de uma das imagens sendo que os dados são listas de valores RBG de uma imagem 30x30
x_train_mod = np.reshape(x_train,(8273,30,30,3))
print(np.shape(x_train))
print(np.shape(x_train_mod))

for i in range(1,5):
    print("Showing image {}, which is qualified as {}".format(i, y_train[i]))
    plt.imshow(x_train_mod[i]);
    plt.show()

# from colorsys import hsv_to_rgb
# from PIL import Image

# # Convert list to bytes ddd
# for i in range(1,5):
#     img = Image.frombytes("RGB", (30, 30), x_train[i])
#     img.show()
#     img.save('pic1.png')
