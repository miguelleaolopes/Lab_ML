from gc import callbacks
from initialization import *

class model:

    def __init__(self,data_augmentation = False):
        self.model = models.Sequential()
        self.data_augmentation = data_augmentation
        self.layers_defined = False

    def summary(self):
        print('##### Model Summary ##########')
        self.model.summary()

    def compile(self,epoch,calls = None,compiler=2):

        if self.layers_defined:
            if compiler == "adam_bin":
                self.model.compile(optimizer = keras.optimizers.Adam(1e-3),
                                    loss = "binary_crossentropy",
                                    metrics = ['accuracy'])
            
            if compiler == "adam_hinge":
                self.model.compile(optimizer = tf.keras.optimizers.Adam(1e-3),
                                    loss = tf.keras.losses.Hinge(), ## To use Hinge the last activation needs to be a tanh
                                    metrics = ['accuracy'])

            if compiler == "sgd_bin":
                self.model.compile(optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3),
                                    loss = "binary_crossentropy",
                                    metrics = ['accuracy'])
            if compiler == "alexnet":
                self.model.compile(
                                loss='binary_crossentropy', 
                                optimizer=tf.optimizers.SGD(learning_rate=0.001), 
                                metrics=['accuracy'])
            
            if compiler == 'InceptionV3':
                from keras.optimizers import RMSprop
                self.model.compile (optimizer = RMSprop(learning_rate=0.0001),
                                    loss = 'binary_crossentropy',
                                    metrics = ['acc'])

            self.history = self.model.fit(x_train, y_train, epochs=epoch, 
                        validation_data=(x_test, y_test),callbacks=calls)
        else:
            "Layers not defined, please define valid layers and compiler first"

    def show_acc_plt(self, name, save_img = False):
        
        # plt.subplot(1, 2, 1)
        plt.plot(self.history.history['accuracy'], label='accuracy')
        plt.plot(self.history.history['val_accuracy'], label = 'val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0.45, 1])
        plt.legend(loc='lower right')
        if save_img == True:
            plt.savefig("models_acc_epo/model_acc_"+ name + ".png")
        # plt.show()

        # plt.subplot(1, 2, 2)
        plt.plot(self.history.history['loss'], label='loss')
        plt.plot(self.history.history['val_loss'], label = 'val_loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.ylim([0.0, plt.ylim()[1]])
        plt.legend(loc='lower right')
        if save_img == True:    
            plt.savefig("models_loss_epo/model_loss_"+ name + ".png")
        # plt.show()

        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['accuracy'], label='accuracy')
        plt.plot(self.history.history['val_accuracy'], label = 'val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0.45, 1])
        plt.legend(loc='lower right')
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['loss'], label='loss')
        plt.plot(self.history.history['val_loss'], label = 'val_loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.ylim([0.0, plt.ylim()[1]])
        plt.legend(loc='lower right')
        plt.subplots_adjust(left=0.06,bottom=0.1,right=0.94,top=0.94,wspace=0.16,hspace=0.2)
        plt.show()

    def show_acc_val(self):
        self.test_loss, self.test_acc = self.model.evaluate(x_test,  y_test, verbose=2)
        print('Total Model Accuracy:', self.test_acc)

        print("Best Validation Accuracy: ", max(self.history.history["val_accuracy"]))

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

        if layers_ind == "with_dropout":
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
        if layers_ind == "without_dropout":
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
            # self.model.add(layers.Dropout(0.2))
            self.model.add(layers.BatchNormalization())

            self.model.add(layers.Dense(128, activation='relu'))
            # self.model.add(layers.Dropout(0.2))
            self.model.add(layers.BatchNormalization())

            self.model.add(layers.Dense(1, activation="sigmoid"))

        if layers_ind == "alexnet":
            self.model.add(layers.Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), activation="relu", input_shape=(30, 30, 3),padding='same'))
            self.model.add(layers.BatchNormalization())
            self.model.add(layers.MaxPool2D(pool_size=(3, 3), strides= (2, 2), padding='same'))
            self.model.add(layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), activation="relu", padding="same"))
            self.model.add(layers.BatchNormalization())
            self.model.add(layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
            self.model.add(layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation="relu", padding="same"))
            self.model.add(layers.BatchNormalization())
            self.model.add(layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation="relu", padding="same"))
            self.model.add(layers.BatchNormalization())
            self.model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation="relu", padding="same"))
            self.model.add(layers.BatchNormalization())
            self.model.add(layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
            self.model.add(layers.Flatten())
            self.model.add(layers.Dense(4096, activation="relu"))
            self.model.add(layers.Dropout(0.5))
            self.model.add(layers.Dense(10, activation="softmax"))

        if layers_ind == 'InceptionV3':
            from keras.applications.inception_v3 import InceptionV3
            
            print('Model only avaiable for images with sizes above 75x75')
            #Replace self.model
            self.model = InceptionV3(
                input_shape = (30,30,3),
                include_top = False, #Leave out last connected layer
                weights = 'imagenet'
            )

            self.model.add(layers.Flatten()(self.model.output))
            self.model.add(layers.Dense(1024, activation='sigmoid'))
            self.model.add(layers.Dropout(0.2))
            self.model.add(layers.Dense(1, activation='sigmoid'))




        self.layers_defined = True
    
