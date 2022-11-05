import numpy as np
from sklearn.feature_extraction.image import reconstruct_from_patches_2d
from matplotlib import pyplot as plt

x_import = np.reshape(np.load('data/Xtrain_Classification2.npy'),(50700,5,5,3))/255.0
y_import = np.load('data/Ytrain_Classification2.npy')


for i in range(0,np.shape(x_import)[0]-676,676):
    image = reconstruct_from_patches_2d(x_import[i:i+676],(30,30,3))
    plt.imshow(image)
    plt.show()
