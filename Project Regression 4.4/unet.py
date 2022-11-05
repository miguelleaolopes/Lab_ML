import numpy as np
from sklearn.feature_extraction.image import reconstruct_from_patches_2d, extract_patches_2d
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from keras.utils import plot_model, to_categorical
import tensorflow as tf
from tensorflow import keras
import keras.backend as K
from keras import datasets, layers, models
from keras.models import load_model  
from colorsys import hsv_to_rgb
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
from sklearn import metrics


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
    image = reconstruct_from_patches_2d(x_import[i*676:(i+1)*676],(30,30,3))[2:28,2:28]
    y_image = reconstruct_from_patches_2d(y_import[i*676:(i+1)*676],(26,26))
    x_import_reconstructed[i] = image
    y_import_reconstructed[i] = y_image

x_train, x_test, y_train, y_test =  train_test_split(x_import_reconstructed, y_import_reconstructed, test_size=0.2)


def show_images(x,y,index):
    plt.subplot(1,2,1)
    plt.imshow(x[index])
    plt.subplot(1,2,2)
    plt.imshow(y[index])
    plt.show()

# for i in range (70,74):
#     show_images(x_import_reconstructed,y_import_reconstructed,i)


## ---------------------------------------------
# Functions to build the U_Net

def double_conv_block(x, n_filters):
   # Conv2D then ReLU activation
   x = layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
   # Conv2D then ReLU activation
   x = layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
   return x

def downsample_block(x, n_filters):
   f = double_conv_block(x, n_filters)
   p = layers.MaxPool2D(2)(f)
   p = layers.Dropout(0.2)(p)
   return f, p

def upsample_block(x, conv_features, n_filters):
   # upsample
   x = layers.Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
   # concatenate
   x = layers.concatenate([x, conv_features])
   # dropout
   x = layers.Dropout(0.2)(x)
   # Conv2D twice with ReLU activation
   x = double_conv_block(x, n_filters)
   return x