# Import Libraries
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import plot_model, to_categorical
import tensorflow as tf
from tensorflow import keras
import keras.backend as K
from keras import datasets, layers, models
from keras.models import load_model  
from colorsys import hsv_to_rgb
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score


# Import lists
x_import = np.load('data/Xtrain_Classification2.npy')
y_import = np.load('data/Ytrain_Classification2.npy')

# Input and Classes
image_size = (5, 5)
input_shape = image_size + (3,)

num_classes = 3
if num_classes==2:
    output_units = 1
    output_activation = "sigmoid"
else:
    output_units = num_classes
    output_activation = "softmax"

# Reshape and One-Hot Encoding 
x_import =  np.reshape(x_import, (np.shape(x_import)[0],) + input_shape)
x_import = x_import/255.0
y_import_onehot = to_categorical(y_import)
x_train, x_test, y_train, y_test =  train_test_split(x_import, y_import_onehot, test_size=0.2)


# Basic Funtions

def show_images(x,y,index):
    print("Showing image {}, which is qualified as {}".format(index, y[index]))
    plt.imshow(x[index])
    plt.show()

for i in range(1000): show_images(x_import,y_import,i)


def convert_npy_to_image(x,y):
    for i in range(np.shape(x)[0]):
        img = Image.frombytes("RGB", (30, 30), x[i])
        if y[i] == 1: #eyespot
            img.save('images/eyespot/pic{}.png'.format(i))
        else:
            img.save('images/spot/pic{}.png'.format(i))

    print('done')

# Create Balanced Accuracy

class BalancedSparseCategoricalAccuracy(keras.metrics.SparseCategoricalAccuracy):
    def __init__(self, name='balanced_sparse_categorical_accuracy', dtype=None):
        super().__init__(name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_flat = y_true
        if y_true.shape.ndims == y_pred.shape.ndims:
            y_flat = tf.squeeze(y_flat, axis=[-1])
        y_true_int = tf.cast(y_flat, tf.int32)

        cls_counts = tf.math.bincount(y_true_int)
        cls_counts = tf.math.reciprocal_no_nan(tf.cast(cls_counts, self.dtype))
        weight = tf.gather(cls_counts, y_true_int)
        return super().update_state(y_true, y_pred, sample_weight=weight)