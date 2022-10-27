from initialization import *
from model import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


my_callbacks = [
    tf.keras.callbacks.CSVLogger("run/training.log", separator=",", append=False),
    tf.keras.callbacks.EarlyStopping(patience=50,verbose=1,monitor='val_loss'), #change to 50 in the end 
    # tf.keras.callbacks.TensorBoard(log_dir='./logs'),
    tf.keras.callbacks.ModelCheckpoint(filepath='run/model_best.h5', monitor='val_accuracy', mode='max', save_best_only=True)
]

m2 = model()
m2.layers("without_dropout")
print('\n\n\n##### End of tensorflow rant ##########\n\n\n')
m2.summary()
m2.compile(epoch=100,calls=my_callbacks,compiler = "bin_adam") 
m2.show_acc_plt("without_dropout",save_img = True)
m2.show_acc_val()

# m3 = model()
# m3.layers('alexnet')
# print('\n\n\n##### End of tensorflow rant ##########\n\n\n')
# m3.summary()
# m3.compile(epoch=150,calls=my_callbacks,compiler = 'alexnet') 
# m3.show_acc_plt('alexnet',save_img = True)
# m3.show_acc_val()