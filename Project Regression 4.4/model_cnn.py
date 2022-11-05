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

    def compile(self,epoch,calls = None,compiler="adam_bal"):
        
        if self.layers_defined:
            if compiler == "adam_bal":
                self.model.compile(optimizer = tf.keras.optimizers.Adam(1e-3),
                                    loss = tf.keras.losses.CategoricalCrossentropy(),
                                    metrics = ['accuracy']) #,balanced_accuracy])
            if compiler == "sgd_bal":
                self.model.compile(optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3),
                                    loss = tf.keras.losses.CategoricalCrossentropy(),
                                    metrics = ['accuracy'])#,balanced_accuracy])
            
            if self.data_augmentation == True:
                train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=5,  # rotation
                                # zoom_range=0.2,  # zoom
                                horizontal_flip=True) # horizontal flip                             
                self.history = self.model.fit(train_datagen.flow(x_train, y_train), epochs=epoch, validation_data=(x_test, y_test),callbacks=calls)
            else:
                self.history = self.model.fit(x_train, y_train, epochs=epoch, validation_data=(x_test, y_test),callbacks=calls)
        else:
            "Layers not defined, please define valid layers and compiler first"

    def show_acc_plt(self, name, save_img = False):
        
        plt.plot(self.history.history['accuracy'], label='accuracy')
        plt.plot(self.history.history['val_accuracy'], label = 'val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0.45, 1])
        plt.legend(loc='lower right')
        if save_img == True:
            plt.savefig("models_acc_epo/model_acc_"+ name + ".png")
        # plt.show()

        plt.plot(self.history.history['loss'], label='loss')
        plt.plot(self.history.history['val_loss'], label = 'val_loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.ylim([0.0, plt.ylim()[1]])
        plt.legend(loc='lower right')
        if save_img == True:    
            plt.savefig("models_loss_epo/model_loss_"+ name + ".png")
        # plt.show()
        
        # plt.plot(self.history.history['balanced_sparse_categorical_accuracy'], label='balanced_sparse_categorical_accuracy')
        # plt.plot(self.history.history['val_balanced_sparse_categorical_accuracy'], label = 'val_balanced_sparse_categorical_accuracy')
        # plt.xlabel('Epoch')
        # plt.ylabel('Balanced Sparse Categorical Accuracy')
        # plt.ylim([0.0, plt.ylim()[1]])
        # plt.legend(loc='lower right')
        # if save_img == True:
        #     plt.savefig("models_BAAC_epo/model_BAAC_"+ name + ".png")
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
        # plt.subplot(1, 3, 3)
        # plt.plot(self.history.history['balanced_sparse_categorical_accuracy'], label='balanced_sparse_categorical_accuracy')
        # plt.plot(self.history.history['val_balanced_sparse_categorical_accuracy'], label = 'val_balanced_sparse_categorical_accuracy')
        # plt.xlabel('Epoch')
        # plt.ylabel('F1 Score')
        # plt.ylim([0.0, plt.ylim()[1]])
        # plt.legend(loc='lower right')
        # plt.subplots_adjust(left=0.06,bottom=0.1,right=0.94,top=0.94,wspace=0.16,hspace=0.2)
        plt.show()

    def show_acc_val(self):

        model_loc = './run/model_best.h5'

        # self.best_model = load_model(model_loc, custom_objects={"BalancedSparseCategoricalAccuracy": BalancedSparseCategoricalAccuracy}) 

        # self.y_pred = np.zeros(np.shape(x_test)[0])
        # self.y_pred = self.best_model.predict(x_test)

        # self.y_pred_bin = np.reshape(np.rint(self.y_pred),(np.shape(self.y_pred)[0]))

        # print("=== Classification Report ===")
        # print(classification_report(y_test, self.y_pred_bin))

        self.test_loss, self.test_acc = self.model.evaluate(x_test,  y_test, verbose=2) # , self.test_BAAC
        print('Final Model Accuracy:', self.test_acc)
        # print('Final Model Balanced Accuracy:', self.test_BAAC)

        print("Best Validation Accuracy: ", max(self.history.history["val_accuracy"]))
        # print("Best BAAC Accuracy: ", max(self.history.history["val_balanced_sparse_categorical_accuracy"]))

    def layers(self,layers_ind=1):
        ## ------------------------------------
        ## task3_model Without Dropout layers
        if layers_ind == "task3_model_dropout":
            self.model.add(layers.Conv2D(32, (3, 3), strides=1, activation='relu', padding="same", input_shape=input_shape))
            self.model.add(layers.MaxPooling2D((2, 2), padding="same"))
            # self.model.add(layers.Dropout(0.2))
            self.model.add(layers.BatchNormalization())

            self.model.add(layers.Conv2D(64, (3, 3), strides=1, activation='relu', padding="same"))
            self.model.add(layers.MaxPooling2D((2, 2), padding="same"))
            # self.model.add(layers.Dropout(0.2))
            self.model.add(layers.BatchNormalization())

            self.model.add(layers.Conv2D(128, (3, 3), strides=1, activation='relu', padding="same"))
            # self.model.add(layers.Dropout(0.2))
            self.model.add(layers.BatchNormalization())

            self.model.add(layers.Flatten())
            
            self.model.add(layers.Dense(256, activation='relu'))
            # self.model.add(layers.Dropout(0.2))
            self.model.add(layers.BatchNormalization())

            self.model.add(layers.Dense(64, activation='relu'))
            # self.model.add(layers.Dropout(0.2))
            self.model.add(layers.BatchNormalization())

            self.model.add(layers.Dense(1, activation="sigmoid"))

        ## task3_model - strides = 1, 32/64/128/256/128 - padding = same
        if layers_ind == "task3_model":
            self.model.add(layers.Conv2D(32, (3, 3), strides=1, activation='relu', padding="same",input_shape=input_shape))
            self.model.add(layers.MaxPooling2D((2, 2), padding="same"))
            self.model.add(layers.Dropout(0.2))
            self.model.add(layers.BatchNormalization())

            self.model.add(layers.Conv2D(64, (3, 3), strides=1, activation='relu', padding="same"))
            self.model.add(layers.MaxPooling2D((2, 2), padding="same"))
            self.model.add(layers.Dropout(0.2))
            self.model.add(layers.BatchNormalization())

            self.model.add(layers.Conv2D(128, (3, 3), strides=1, activation='relu', padding="same"))
            self.model.add(layers.Dropout(0.2))
            self.model.add(layers.BatchNormalization())

            self.model.add(layers.Flatten())
            
            self.model.add(layers.Dense(256, activation='relu'))
            self.model.add(layers.Dropout(0.2))
            self.model.add(layers.BatchNormalization())

            self.model.add(layers.Dense(64, activation='relu'))
            self.model.add(layers.Dropout(0.2))
            self.model.add(layers.BatchNormalization())

            self.model.add(layers.Dense(3, activation="softmax"))

        ## MLP - com 1Conv
        if layers_ind == "new_1":
            self.model.add(layers.Conv2D(32, (3, 3), strides=2, activation='relu', padding="valid",input_shape=input_shape))
            self.model.add(layers.MaxPooling2D((2, 2), padding="valid"))
            self.model.add(layers.Dropout(0.2))
            self.model.add(layers.BatchNormalization())

            # self.model.add(layers.Conv2D(64, (3, 3), strides=2, activation='relu', padding="valid"))
            # self.model.add(layers.MaxPooling2D((2, 2), padding="valid"))
            # self.model.add(layers.Dropout(0.2))
            # self.model.add(layers.BatchNormalization())

            # self.model.add(layers.Conv2D(128, (3, 3), strides=2, activation='relu', padding="valid"))
            # self.model.add(layers.Dropout(0.2))
            # self.model.add(layers.BatchNormalization())

            self.model.add(layers.Flatten())
            
            self.model.add(layers.Dense(256, activation='relu'))
            self.model.add(layers.Dropout(0.2))
            self.model.add(layers.BatchNormalization())

            self.model.add(layers.Dense(128, activation='relu'))
            self.model.add(layers.Dropout(0.2))
            self.model.add(layers.BatchNormalization())

            self.model.add(layers.Dense(3, activation="softmax"))

        ## ------------------------------------
        ## MLP - 512,256 - with dropout 0.2
        if layers_ind == "new_2":
            self.model.add(layers.Flatten(input_shape=input_shape))

            self.model.add(layers.Dense(512, activation='relu'))
            self.model.add(layers.Dropout(0.2))
            self.model.add(layers.BatchNormalization())

            self.model.add(layers.Dense(256, activation='relu'))
            self.model.add(layers.Dropout(0.2))
            self.model.add(layers.BatchNormalization())

            self.model.add(layers.Dense(3, activation="softmax"))
        
        ## ------------------------------------
        ## MLP - 512,256 - with dropout 0.2
        if layers_ind == "new_3": 
            self.model.add(layers.Flatten(input_shape=(30, 30, 3)))
            
            self.model.add(layers.Dense(256, activation='relu'))
            self.model.add(layers.Dropout(0.2))
            self.model.add(layers.BatchNormalization())

            self.model.add(layers.Dense(64, activation='relu'))
            self.model.add(layers.Dropout(0.2))
            self.model.add(layers.BatchNormalization())

            self.model.add(layers.Dense(3, activation="softmax"))

        ## ------------------------------------
        if layers_ind == "U_Net":
    # inputs
            inputs = layers.Input(shape=(26,26,3))
            # encoder: contracting path - downsample
            # 1 - downsample
            f1, p1 = downsample_block(inputs, 64)
            # 2 - downsample
            f2, p2 = downsample_block(p1, 128)
            # 3 - downsample
            f3, p3 = downsample_block(p2, 256)
            # 4 - downsample
            f4, p4 = downsample_block(p3, 512)
            # 5 - bottleneck
            bottleneck = double_conv_block(p4, 1024)
            # decoder: expanding path - upsample
            # 6 - upsample
            u6 = upsample_block(bottleneck, f4, 512)
            # 7 - upsample
            u7 = upsample_block(u6, f3, 256)
            # 8 - upsample
            u8 = upsample_block(u7, f2, 128)
            # 9 - upsample
            u9 = upsample_block(u8, f1, 64)
            # outputs
            outputs = layers.Conv2D(3, 1, padding="same", activation = "softmax")(u9)
            # unet model with Keras Functional API
            self.model = tf.keras.Model(inputs, outputs, name="U-Net")

        self.layers_defined = True
    
