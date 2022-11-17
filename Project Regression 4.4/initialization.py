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
from sklearn import metrics

# Import lists
x_import = np.load('data/Xtrain_Classification2.npy')
y_import = np.load('data/Ytrain_Classification2.npy')


print(np.shape(x_import))

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

# for i in range(1000): show_images(x_import,y_import,i)


def convert_npy_to_image(x,y):
    for i in range(np.shape(x)[0]):
        img = Image.frombytes("RGB", (30, 30), x[i])
        if y[i] == 1: #eyespot
            img.save('images/eyespot/pic{}.png'.format(i))
        else:
            img.save('images/spot/pic{}.png'.format(i))

    print('done')

# Create Balanced Accuracy
## Not Working
def balanced_accuracy(y_true,y_pred, sample_weight=None, adjusted=False):
    y_true= y_true.numpy()
    y_pred= y_pred.numpy()
    y_pred2=[np.argmax(i) for i in y_pred]
    y_true2=[np.argmax(i) for i in y_true]
    # C = confusion_matrix(y_true2, y_pred2, sample_weight=sample_weight)
    # with np.errstate(divide="ignore", invalid="ignore"):
    #     per_class = np.diag(C) / C.sum(axis=1)
    #     per_class = per_class[~np.isnan(per_class)]
    # score = np.mean(per_class)
    # if adjusted:
    #     n_classes = len(per_class)
    #     chance = 1 / n_classes
    #     score -= chance
    #     score /= 1 - chance
    # return score
    return balanced_accuracy_score(y_true2,y_pred2)

# def balanced_accuracy_2(y_true, y_pred): #taken from old keras source code
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
#     predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
#     precision = true_positives / (predicted_positives + K.epsilon())
#     recall = true_positives / (possible_positives + K.epsilon())
#     f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
#     return f1_val



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

