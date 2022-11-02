# Import Libraries
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import plot_model
import tensorflow as tf
from tensorflow import keras
import keras.backend as K
# from tensorflow_addons.metrics import F1Score
from keras import datasets, layers, models
from keras.models import load_model 
from colorsys import hsv_to_rgb
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix


# Import lists
x_import = np.load('data/Xtest_Classification1.npy')
y_import = np.load('data/y_pred_1.npy')

x_import = np.reshape(x_import,(1367,30,30,3))

# x_import = np.load('data/Xtrain_Classification1.npy')
# y_import = np.load('data/Ytrain_Classification1.npy')

# x_import = np.reshape(x_import,(8273,30,30,3))

def show_images(x,y,index):
    print("Showing image {}, which is qualified as Migardo - {}".format(index, y[index]))
    plt.imshow(x[index])
    plt.show()
for i in [34, 50, 76, 88, 90, 91, 102, 112, 124, 133, 142, 148, 153, 174, 179, 191, 194, 196, 200, 204, 226, 228, 232, 256, 266, 270, 292, 305, 317, 341, 360, 380, 391, 398, 400, 429, 431, 438, 455, 484, 486, 490, 507, 522, 543, 548, 550, 563, 567, 571, 575, 578, 589, 611, 642, 658, 670, 672, 684, 687, 724, 744, 767, 773, 778, 797, 902, 916, 926, 963, 965, 968, 994, 999, 1001, 1011, 1027, 1035, 1070, 1079, 1080, 1090, 1106, 1117, 1119, 1140, 1144, 1154, 1157, 1167, 1236, 1242, 1245, 1262, 1276, 1277, 1295, 1305, 1324, 1353, 1359]:
    show_images(x_import,y_import,i)