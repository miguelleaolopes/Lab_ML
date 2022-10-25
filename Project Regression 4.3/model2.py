from initialization import *

class model2:

    def __init__(self):
        self.model = models.Sequential()


    def create_model(self):
        self.model.add(layers.Conv2D(32, 3, strides=2, activation='relu', padding="same"))
        self.model.add(layers.MaxPooling2D((2, 2), padding="same"))
        self.model.add(layers.Dropout(0.2))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Conv2D(64, 3, strides=2, activation='relu', padding="same"))
        self.model.add(layers.MaxPooling2D((2, 2), padding="same"))
        self.model.add(layers.Dropout(0.2))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Conv2D(128, 3, strides=2, activation='relu', padding="same"))
        self.model.add(layers.Dropout(0.2))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Flatten())
        self.model.add(layers.MaxPooling2D(3, strides=2, padding="same"))
        self.model.add(layers.Dense(256, activation='relu'))
        self.model.add(layers.Dropout(0.2))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Dense(128, activation='relu'))
        self.model.add(layers.Dropout(0.2))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Dense(1, activation="sigmoid"))


    def summary(self):
        print('##### Model Summary ##########')
        self.model.summary()

    def compile(self):
        self.model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

        self.history = self.model.fit(x_train, y_train, epochs=10, 
                    validation_data=(x_test, y_test))

    def show_acc_plt(self):
        plt.plot(self.history.history['accuracy'], label='accuracy')
        plt.plot(self.history.history['val_accuracy'], label = 'val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0.5, 1])
        plt.legend(loc='lower right')
        plt.show()


    def show_acc_val(self):
        self.test_loss, self.test_acc = self.model.evaluate(x_test,  y_test, verbose=2)
        print('Total Model Accuracy:', self.test_acc)    
    