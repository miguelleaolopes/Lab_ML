from initialization import *
from model_cnn import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

my_callbacks = [
    tf.keras.callbacks.CSVLogger("run/training.log", separator=",", append=False),
    tf.keras.callbacks.EarlyStopping(patience=50,verbose=1,monitor='val_loss'),
    tf.keras.callbacks.ModelCheckpoint(filepath='run/model_best.h5', monitor='val_accuracy', mode='max', save_best_only=True)
]

model_used = "LeNet"
model_used = "U_Net"
model_used = "new_4"
model_used = "new_3"
model_used = "task3_model_dropout"
model_used = "task3_model"
model_used = "new_1"
model_used = "new_2"

m2 = model(data_augmentation=False) #data_augmentation=data_augmentations
m2.layers(model_used)
print('\n\n\n##### End of tensorflow rant ##########\n\n\n')
m2.summary()
m2.compile(epoch=10,calls=my_callbacks,compiler = "adam_bal")  #adam_bin
m2.show_acc_plt(model_used,save_img = False)
m2.show_acc_val()

