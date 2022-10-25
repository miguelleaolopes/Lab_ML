# Import Libraries
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import plot_model
import tensorflow as tf
from tensorflow import keras
from keras import datasets, layers, models


# Import lists
x_import = np.load('data/Xtrain_Classification1.npy')
y_import = np.load('data/Ytrain_Classification1.npy')

x_import = np.reshape(x_import,(8273,30,30,3))
x_train, x_test, y_train, y_test =  train_test_split(x_import, y_import, test_size=0.2)
x_train, x_test = x_train / 255.0, x_test / 255.0


# Basic Funtions

def show_images(x,y,index):
    print("Showing image {}, which is qualified as {}".format(index, y[index]))
    plt.imshow(x[index])
    plt.show()


    