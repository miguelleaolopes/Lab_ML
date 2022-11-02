from initialization import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

my_callbacks = [
    tf.keras.callbacks.ModelCheckpoint(filepath='run_final/model_best.h5', monitor='f1_score', mode='max', save_best_only=True)
]

model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), strides=1, activation='relu', padding="same",input_shape=(30, 30, 3)))
model.add(layers.MaxPooling2D((2, 2), padding="same"))
model.add(layers.Dropout(0.2))
model.add(layers.BatchNormalization())

model.add(layers.Conv2D(64, (3, 3), strides=1, activation='relu', padding="same"))
model.add(layers.MaxPooling2D((2, 2), padding="same"))
model.add(layers.Dropout(0.2))
model.add(layers.BatchNormalization())

model.add(layers.Conv2D(128, (3, 3), strides=1, activation='relu', padding="same"))
model.add(layers.Dropout(0.2))
model.add(layers.BatchNormalization())

model.add(layers.Flatten())

model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.BatchNormalization())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.BatchNormalization())

model.add(layers.Dense(1, activation="sigmoid"))

print('\n\n\n##### End of tensorflow rant ##########\n\n\n')

print('##### Model Summary ##########')
model.summary()

print('##### Compiling ##########')
model.compile(
            optimizer = tf.keras.optimizers.Adam(1e-3),
            loss = "binary_crossentropy",
            metrics = ['accuracy'])                         ## ADD BAAC


train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                                rotation_range=5,  # rotation
                                zoom_range=0.2,  # zoom
                                horizontal_flip=True) # horizontal flip   

history = model.fit(
                    train_datagen.flow(x_import, y_import), 
                    epochs=30, 
                    #validation_data=(x_test, y_test),             
                    callbacks=my_callbacks
                    )

print('##### Model Created #####')
model_loc = './run_final/model_best.h5'
# best_model = load_model(model_loc, custom_objects={"F1_Score": F1_Score,"get_f1": get_f1}) 
print("Best F1Score Accuracy: ", max(history.history["f1_score"]))

X_TEST = np.load('data/Xtest_Classification1.npy')
X_TEST = np.reshape(X_TEST,(np.shape(X_TEST)[0],30,30,3)) / 255.0
print('X_TEST Shape:',np.shape(X_TEST))
y_pred = model.predict(X_TEST)
y_pred_bin = np.reshape(np.rint(y_pred),(np.shape(y_pred)[0]))
np.save('data/y_pred.npy', y_pred_bin)
print ('Y prediction saved')
