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

# model_used = "without_dropout"
model_used = "with_dropout_2"

m2 = model()
m2.layers(model_used)
print('\n\n\n##### End of tensorflow rant ##########\n\n\n')
m2.summary()
m2.compile(epoch=150,calls=my_callbacks,compiler = "adam_bin") 
m2.show_acc_plt(model_used,save_img = True)
m2.show_acc_val()

