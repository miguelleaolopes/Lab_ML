import numpy as np
from sklearn.feature_extraction.image import reconstruct_from_patches_2d, extract_patches_2d
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
 


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

x_train, x_val, y_train, y_val =  train_test_split(x_import_reconstructed, y_import_reconstructed, test_size=0.2)

print(np.shape(x_train))
print(np.shape(y_train))


def double_conv_block(x, n_filters):
   # Conv2D then ReLU activation
   x = layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
   # Conv2D then ReLU activation
   x = layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
   return x

def downsample_block(x, n_filters):
   f = double_conv_block(x, n_filters)
   p = layers.MaxPool2D(2)(f)
   p = layers.Dropout(0.3)(p)
   return f, p

def upsample_block(x, conv_features, n_filters):
   # upsample
   x = layers.Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
   # concatenate
   x = layers.concatenate([x, conv_features])
   # dropout
   x = layers.Dropout(0.3)(x)
   # Conv2D twice with ReLU activation
   x = double_conv_block(x, n_filters)
   return x

def build_unet_model():
   inputs = layers.Input(shape=(26,26,3))
   # encoder: contracting path - downsample
   # 1 - downsample
   f1, p1 = downsample_block(inputs, 64)
   # 2 - downsample
   f2, p2 = downsample_block(p1, 128)
   # 3 - downsample
   f3, p3 = downsample_block(p2, 256)
   # 4 - downsample
   f4, p4 = downsample_block(p3, 512)
   # 5 - bottleneck
   bottleneck = double_conv_block(p4, 1024)
   # decoder: expanding path - upsample
   # 6 - upsample
   u6 = upsample_block(bottleneck, f4, 512)
   # 7 - upsample
   u7 = upsample_block(u6, f3, 256)
   # 8 - upsample
   u8 = upsample_block(u7, f2, 128)
   # 9 - upsample
   u9 = upsample_block(u8, f1, 64)
   # outputs
   outputs = layers.Conv2D(3, 1, padding="same", activation = "softmax")(u9)
   # unet model with Keras Functional API
   unet_model = tf.keras.Model(inputs, outputs, name="U-Net")
   return unet_model

unet_model = build_unet_model()

unet_model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss="sparse_categorical_crossentropy",
                  metrics="accuracy")

BATCH_SIZE = 5
NUM_EPOCHS = 20
STEPS_PER_EPOCH = np.lenght(x_train)[0] // BATCH_SIZE
VAL_SUBSPLITS = 5
VALIDATION_STEPS = np.lenght(y_train)[0] // BATCH_SIZE // VAL_SUBSPLITS
model_history = unet_model.fit(x_train,
                              epochs=NUM_EPOCHS,
                              steps_per_epoch=STEPS_PER_EPOCH,
                              validation_steps=VALIDATION_STEPS,
                              validation_data=y_train)