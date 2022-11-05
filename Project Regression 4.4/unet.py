import numpy as np
from sklearn.feature_extraction.image import reconstruct_from_patches_2d, extract_patches_2d
from matplotlib import pyplot as plt

x_import = np.load('data/Xtrain_Classification2.npy')/255.0
y_import = np.load('data/Ytrain_Classification2.npy')
x_final_test = np.load('data/Xtest_Classification2.npy')/255.0

patch_size = (5, 5)
input_shape = patch_size + (3,)
x_import = np.reshape(x_import,(50700,5,5,3))
y_import = np.reshape(y_import,(50700,1,1))
x_import_reconstructed = np.zeros((75,26,26,3))
y_import_reconstructed = np.zeros((75,26,26))

for i in range(0,74):
    image = reconstruct_from_patches_2d(x_import[i*676:(i+1)*675],(30,30,3))[2:28,2:28]
    y_image = reconstruct_from_patches_2d(y_import[i*676:(i+1)*675],(26,26))
    x_import_reconstructed[i] = image
    y_import_reconstructed[i] = y_image

print (np.shape(x_import_reconstructed))

# for i in range(np.shape(x_import_reconstructed)[0]-1,0,-1):
for i in range(0,np.shape(x_import_reconstructed)[0]-1):
    plt.imshow(x_import_reconstructed[i])
    plt.show()
    plt.imshow(y_import_reconstructed[i])
    plt.show()
