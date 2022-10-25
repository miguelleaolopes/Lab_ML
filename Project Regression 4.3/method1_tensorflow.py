from initialization import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

image_size = (30, 30)
batch_size = 32

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "images",
    validation_split=0.2,
    subset="training",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "images",
    validation_split=0.2,
    subset="validation",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)

# Augmentation of data by introducing some random rotation

data_augmentation = keras.Sequential()


def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    # Image augmentation block
    x = data_augmentation(inputs)

    # Entry block
    x = layers.Rescaling(1.0 / 255)(x)

    x = layers.Conv2D(32, 3, strides=2, activation='relu', padding="same")(x)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(64, 3, strides=2, activation='relu', padding="same")(x)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(128, 3, strides=2, activation='relu', padding="same")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Flatten()(x)

    # x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.BatchNormalization()(x)

    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    outputs = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)

# Rescaling

inputs = keras.Input(shape=image_size + (3,))
x = data_augmentation(inputs)
x = layers.Rescaling(1./255)(x)

train_ds = train_ds.prefetch(buffer_size=32)
val_ds = val_ds.prefetch(buffer_size=32)

# Run the model

model = make_model(input_shape=image_size + (3,), num_classes=2)
keras.utils.plot_model(model, show_shapes=True)
print(model.summary())
epochs = 50

callbacks = [
    keras.callbacks.ModelCheckpoint("save_epoch_1/save_at_{epoch}.h5"),
]
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    # loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)
history = model.fit(
    train_ds, epochs=epochs, callbacks=callbacks, validation_data=val_ds,
)

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.savefig("method_1_acc_epo.png")
plt.show()
# test_loss, test_acc = model.evaluate(x_import,  y_import, verbose=2)
# print(test_acc)