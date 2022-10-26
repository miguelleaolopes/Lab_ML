from initialization import *
from model1 import *
from model2 import *

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# m1 = model1()
# m1.create_model()
# print('\n\n\n##### End of tensorflow rant ##########\n\n\n')
# m1.summary()
# m1.compile()
# m1.show_acc_plt(save_img = True)
# m1.show_acc_val()

my_callbacks = [
    tf.keras.callbacks.CSVLogger("run/training.log", separator=",", append=False),
    tf.keras.callbacks.EarlyStopping(patience=10,verbose=1,monitor='val_loss'), #change to 50 in the end 
    tf.keras.callbacks.TensorBoard(log_dir='./logs'),
    tf.keras.callbacks.ModelCheckpoint(filepath='run/model_best.h5', monitor='val_accuracy', mode='max', save_best_only=True)
]

m2 = model2()
m2.create_model()
print('\n\n\n##### End of tensorflow rant ##########\n\n\n')
m2.summary()
m2.compile(epoch=50,calls=my_callbacks)
m2.show_acc_plt(save_img = True)
m2.show_acc_val()