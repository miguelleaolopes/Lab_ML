from gc import callbacks
from initialization import *

class model:

    def __init__(self,data_augmentation = False):
        self.model = models.Sequential()
        self.data_augmentation = data_augmentation
        self.layers_defined = False

    def create_model(self):
        self.model.add(layers.Conv2D(64, 3, strides=2, activation='relu', padding="same", input_shape=(30, 30, 3)))
        self.model.add(layers.MaxPooling2D((2, 2), padding="same"))
        self.model.add(layers.Dropout(0.2))
        self.model.add(layers.BatchNormalization())

        self.model.add(layers.Conv2D(128, 3, strides=2, activation='relu', padding="same"))
        self.model.add(layers.MaxPooling2D((2, 2), padding="same"))
        self.model.add(layers.Dropout(0.2))
        self.model.add(layers.BatchNormalization())

        self.model.add(layers.Conv2D(256, 3, strides=2, activation='relu', padding="same"))
        self.model.add(layers.Dropout(0.2))
        self.model.add(layers.BatchNormalization())

        self.model.add(layers.Flatten())
        
        self.model.add(layers.Dense(512, activation='relu'))
        self.model.add(layers.Dropout(0.2))
        self.model.add(layers.BatchNormalization())

        self.model.add(layers.Dense(256, activation='relu'))
        self.model.add(layers.Dropout(0.2))
        self.model.add(layers.BatchNormalization())

        self.model.add(layers.Dense(1, activation="sigmoid"))

    def summary(self):
        print('##### Model Summary ##########')
        self.model.summary()

    def compile(self,epoch,calls = None,compiler=2):

        if self.layers_defined:
            if compiler == 1:
                self.model.compile(
                            optimizer = 'adam',
                            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                            metrics = ['accuracy'])
            if compiler == 2:
                self.model.compile(optimizer = keras.optimizers.Adam(1e-3),
                                    loss = "binary_crossentropy",
                                    metrics = ['accuracy'])
            
            if compiler == 3:
                self.model.compile(optimizer = tf.keras.optimizers.Adam(1e-3),
                                    loss = tf.keras.losses.Hinge(), ## To use Hinge the last activation needs to be a tanh
                                    metrics = ['accuracy'])

            if compiler == 4:
                self.model.compile(optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3),
                                    loss = "binary_crossentropy",
                                    metrics = ['accuracy'])

            self.history = self.model.fit(x_train, y_train, epochs=epoch, 
                        validation_data=(x_test, y_test),callbacks=calls)
        else:
            "Layers not defined, please define valid layers and compiler first"

    def show_acc_plt(self, save_img = False):
        plt.plot(self.history.history['accuracy'], label='accuracy')
        plt.plot(self.history.history['val_accuracy'], label = 'val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0.45, 1])
        plt.legend(loc='lower right')
        if save_img == True:
            plt.savefig("models_acc_epo/model2_acc_epo.png")
        plt.show()

    def show_acc_val(self):
        self.test_loss, self.test_acc = self.model.evaluate(x_test,  y_test, verbose=2)
        print('Total Model Accuracy:', self.test_acc)

    def layers(self,layers_ind=1):
        if layers_ind == 1:
            self.model.add(layers.Conv2D(30, (3, 3), activation='relu', input_shape=(30, 30, 3)))
            self.model.add(layers.MaxPooling2D((2, 2)))
            self.model.add(layers.Conv2D(60, (3, 3), activation='relu'))
            self.model.add(layers.MaxPooling2D((2, 2)))
            self.model.add(layers.Conv2D(60, (3, 3), activation='relu'))
            self.model.add(layers.Flatten())
            self.model.add(layers.Dense(60, activation='relu'))
            self.model.add(layers.Dense(10))

        ## With Dropout layers
        if layers_ind == 2:
            self.model.add(layers.Conv2D(32, (3, 3), strides=2, activation='relu', padding="same", input_shape=(30, 30, 3)))
            self.model.add(layers.MaxPooling2D((2, 2), padding="same"))
            self.model.add(layers.Dropout(0.2))
            self.model.add(layers.BatchNormalization())

            self.model.add(layers.Conv2D(64, (3, 3), strides=2, activation='relu', padding="same"))
            self.model.add(layers.MaxPooling2D((2, 2), padding="same"))
            self.model.add(layers.Dropout(0.2))
            self.model.add(layers.BatchNormalization())

            self.model.add(layers.Conv2D(128, (3, 3), strides=2, activation='relu', padding="same"))
            self.model.add(layers.Dropout(0.2))
            self.model.add(layers.BatchNormalization())

            self.model.add(layers.Flatten())
            
            self.model.add(layers.Dense(256, activation='relu'))
            self.model.add(layers.Dropout(0.2))
            self.model.add(layers.BatchNormalization())

            self.model.add(layers.Dense(128, activation='relu'))
            self.model.add(layers.Dropout(0.2))
            self.model.add(layers.BatchNormalization())

            self.model.add(layers.Dense(1, activation="sigmoid"))
                
        ## Without Dropout layers
        if layers_ind == 3:
            self.model.add(layers.Conv2D(32, (3, 3), strides=2, activation='relu', padding="same", input_shape=(30, 30, 3)))
            self.model.add(layers.MaxPooling2D((2, 2), padding="same"))
            # self.model.add(layers.Dropout(0.2))
            self.model.add(layers.BatchNormalization())

            self.model.add(layers.Conv2D(64, (3, 3), strides=2, activation='relu', padding="same"))
            self.model.add(layers.MaxPooling2D((2, 2), padding="same"))
            # self.model.add(layers.Dropout(0.2))
            self.model.add(layers.BatchNormalization())

            self.model.add(layers.Conv2D(128, (3, 3), strides=2, activation='relu', padding="same"))
            # self.model.add(layers.Dropout(0.2))
            self.model.add(layers.BatchNormalization())

            self.model.add(layers.Flatten())
            
            self.model.add(layers.Dense(256, activation='relu'))
            self.model.add(layers.Dropout(0.2))
            self.model.add(layers.BatchNormalization())

            self.model.add(layers.Dense(128, activation='relu'))
            self.model.add(layers.Dropout(0.2))
            self.model.add(layers.BatchNormalization())

            self.model.add(layers.Dense(1, activation="sigmoid"))

        self.layers_defined = True
    


