import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.image import reconstruct_from_patches_2d

x_import = np.load('data/Xtrain_Classification2.npy')/255
y_import = np.load('data/Ytrain_Classification2.npy')

patch_size = (5, 5)
input_shape = patch_size + (3,)

x_import_reshaped =  np.reshape(x_import, (np.shape(x_import)[0],) + input_shape)

for i in range(0,np.shape(x_import)[0]-676,676):
    plt.imshow(reconstruct_from_patches_2d(x_import_reshaped[i:i+676,],(26,26,3)))
    plt.show()