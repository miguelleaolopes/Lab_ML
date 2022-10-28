from initialization import *
from model import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# data_augmentations = keras.Sequential([
#     tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
#     tf.keras.layers.experimental.preprocessing.RandomRotation(0.1,input_shape = (30,30,3)),
#     tf.keras.layers.experimental.preprocessing.RandomZoom(0.1)
# ])


my_callbacks = [
    tf.keras.callbacks.CSVLogger("run/training.log", separator=",", append=False),
    tf.keras.callbacks.EarlyStopping(patience=50,verbose=1,monitor='val_loss'), #change to 50 in the end 
    # tf.keras.callbacks.TensorBoard(log_dir='./logs'),
    tf.keras.callbacks.ModelCheckpoint(filepath='run/model_best.h5', monitor='val_accuracy', mode='max', save_best_only=True)
]

model_used = "without_dropout"
model_used = "with_dropout"
model_used = "LeNet"
model_used = "with_dropout_2"
model_used = "with_dropout_3"
model_used = "with_dropout_4"
model_used = "Test1"

m2 = model(data_augmentation=False) #data_augmentation=data_augmentations
m2.layers(model_used)
print('\n\n\n##### End of tensorflow rant ##########\n\n\n')
m2.summary()
m2.compile(epoch=10,calls=my_callbacks,compiler = "adam_bin")  #adam_bin
m2.show_acc_plt(model_used,save_img = True)
m2.show_acc_val()

