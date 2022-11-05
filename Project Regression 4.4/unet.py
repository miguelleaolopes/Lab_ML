import numpy as np
from sklearn.feature_extraction.image import reconstruct_from_patches_2d
from matplotlib import pyplot as plt

x_import = np.reshape(np.load('data/Xtrain_Classification2.npy'),(50700,5,5,3))/255.0
y_import = np.load('data/Ytrain_Classification2.npy')
print(np.shape(y_import))

x_import_reconstructed = np.zeros((75,26,26,3))

x_index = 0
for i in range(0,50699-676,676):
    image = reconstruct_from_patches_2d(x_import[i:i+676],(30,30,3))[2:28,2:28]
    x_import_reconstructed[int(x_index)] = image
    x_index+=1

print (np.shape(x_import_reconstructed))

for i in range(np.shape(x_import_reconstructed)[0]-1,0,-1):
    plt.imshow(x_import_reconstructed[i])
    plt.show()
