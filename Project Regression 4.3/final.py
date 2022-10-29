from initialization import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

my_callbacks = [
    tf.keras.callbacks.ModelCheckpoint(filepath='run_final/model_best.h5', monitor='f1_score', mode='max', save_best_only=True)
]

model = models.Sequential()

# With Dropout 2
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
model.add(layers.Dense(128, activation='relu'))
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
            metrics = ['accuracy', get_f1, F1_Score()])


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
best_model = load_model(model_loc, custom_objects={"F1_Score": F1_Score,"get_f1": get_f1}) 

X_TEST = np.load('data/Xtest_Classification1.npy')
X_TEST = np.reshape(X_TEST,(np.shape(X_TEST)[0],30,30,3)) / 255.0
print('X_TEST Shape:',np.shape(X_TEST))
y_pred = best_model.predict(X_TEST)
y_pred_bin = np.reshape(np.rint(y_pred),(np.shape(y_pred)[0]))
np.save('data/y_pred.npy', y_pred_bin)
print ('Y prediction saved')


plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.45, 1])
plt.legend(loc='lower right')
plt.savefig("models_acc_epo/model_acc_final.png")
plt.show()

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim([0.0, plt.ylim()[1]])
plt.legend(loc='lower right')
plt.savefig("models_loss_epo/model_loss_final.png")
plt.show()

plt.plot(history.history['val_f1_score'], label = 'val_f1_score')
plt.plot(history.history['f1_score'], label='f1_score')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.ylim([0.0, plt.ylim()[1]])
plt.legend(loc='lower right')
plt.savefig("models_f1score_epo/model_f1score_final.png")
plt.show()

plt.subplot(1, 3, 1)
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.45, 1])
plt.legend(loc='lower right')
plt.subplot(1, 3, 2)
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim([0.0, plt.ylim()[1]])
plt.legend(loc='lower right')
plt.subplots_adjust(left=0.06,bottom=0.1,right=0.94,top=0.94,wspace=0.16,hspace=0.2)
plt.subplot(1, 3, 3)
plt.plot(history.history['f1_score'], label='f1_score')
plt.plot(history.history['val_f1_score'], label = 'val_f1_score')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.ylim([0.0, plt.ylim()[1]])
plt.legend(loc='lower right')
plt.subplots_adjust(left=0.06,bottom=0.1,right=0.94,top=0.94,wspace=0.16,hspace=0.2)
plt.show()




