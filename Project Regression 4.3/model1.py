from initialization import *

class model1:

    def __init__(self) -> None:
        pass
        
    def create_model(self):
        self.model = models.Sequential()
        self.model.add(layers.Conv2D(30, (3, 3), activation='relu', input_shape=(30, 30, 3)))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(60, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(60, (3, 3), activation='relu'))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(60, activation='relu'))
        self.model.add(layers.Dense(10))

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
    