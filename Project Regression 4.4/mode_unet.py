# Import Libraries
import os
from unet import *
class model:
    def __init__(self,data_augmentation = False):
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

        # model_loc = './run/model_best.h5'

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
    


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

m = model(data_augmentation=False) 
m.layers("U_Net")
print('\n\n\n##### End of tensorflow rant ##########\n\n\n')
m.summary()
m.compile(epoch=10,calls=None,compiler = "adam_bal")  
m.show_acc_plt("U_Net",save_img = False)
m.show_acc_val()
